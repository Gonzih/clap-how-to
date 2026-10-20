//! Bundled sentence-transformer embeddings, offline, with native Rust CPU inference.
//!
//! This runs all three attention layers of MiniLM. The 4-bit **storage** format is
//! dequantized to F32 once at load; inference is F32, not an integer kernel.
//! No model downloads, Python, ONNX shared library, cache or build script.
//!
//! ```no_run
//! let encoder = pocket_encoder::Encoder::new()?;
//! let vectors = encoder.embed(&["Send a document", "Transfer a file"])?;
//! assert_eq!(vectors[0].len(), 384);
//! # Ok::<(), pocket_encoder::Error>(())
//! ```
//!
//! Enable the `search` feature for an exact cosine index with compact binary
//! storage. Embedding-only users do not compile the index API.
#[cfg(feature = "search")]
mod index;
#[cfg(feature = "search")]
pub use index::{Index, Neighbor};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};
use tokenizers::{Tokenizer, TruncationParams};

/// Embedding coordinate-space identity. Persist this with indexed vectors.
/// Exact equality, not just equal dimensions, is required before comparison.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Identifies the model and preprocessing contract.
    pub id: String,
    /// SHA-256 of the bundled quantized weights.
    pub revision: String,
    /// Upstream model repository.
    pub upstream_model: String,
    /// Immutable upstream revision.
    pub upstream_revision: String,
    /// SHA-256 of the unchanged tokenizer JSON.
    pub tokenizer_sha256: String,
    /// Number of normalized F32 coordinates per input.
    pub dimensions: usize,
    /// Includes special tokens. Longer inputs are right-truncated.
    pub max_tokens: usize,
    /// Pooling and normalization contract.
    pub pooling: String,
}

/// Metadata is available without loading weights or doing inference.
pub fn model_info() -> ModelInfo {
    let value: serde_json::Value = serde_json::from_str(include_str!("../assets/provenance.json"))
        .expect("bundled provenance");
    ModelInfo {
        id: "pocket-encoder/minilm-l3-q4-g32-v1".into(),
        revision: format!("sha256:{}", value["bundled_sha256"].as_str().unwrap()),
        upstream_model: value["upstream_model"].as_str().unwrap().into(),
        upstream_revision: value["upstream_revision"].as_str().unwrap().into(),
        tokenizer_sha256: value["tokenizer_sha256"].as_str().unwrap().into(),
        dimensions: 384,
        max_tokens: 128,
        pooling: "attention-masked mean; L2 normalized; no query prefix; right truncation".into(),
    }
}

/// Validation or native inference failure. Input text is never included.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Input bounds were exceeded.
    #[error("{0}")]
    Input(&'static str),
    /// Tokenizer/model initialization or inference failed.
    #[error("encoder failed: {0}")]
    Inference(String),
}

struct Inner {
    model: BertModel,
    tokenizer: Tokenizer,
}

/// Cheap clones share immutable weights and tokenizer; CPU only.
#[derive(Clone)]
pub struct Encoder {
    inner: Arc<Inner>,
}

impl Encoder {
    /// Load compiled-in weights once per process. No filesystem or network I/O.
    pub fn new() -> Result<Self, Error> {
        static MODEL: OnceLock<Result<Arc<Inner>, String>> = OnceLock::new();
        MODEL
            .get_or_init(|| load_bundled().map(Arc::new).map_err(|e| e.to_string()))
            .clone()
            .map(|inner| Self { inner })
            .map_err(Error::Inference)
    }

    /// Encode at most 4096 inputs, each at most 64 KiB, at most 4 MiB combined.
    /// Inputs are independently right-truncated to 128 WordPiece tokens, including
    /// special tokens. Use smaller chunks for long documents. Empty text is rejected.
    /// Returns unit-length vectors in the same order; no input text is retained.
    pub fn embed(&self, texts: &[impl AsRef<str>]) -> Result<Vec<Vec<f32>>, Error> {
        if texts.len() > 4096 {
            return Err(Error::Input("at most 4096 texts per call"));
        }
        let mut bytes = 0usize;
        for text in texts {
            let text = text.as_ref();
            if text.trim().is_empty() {
                return Err(Error::Input("text must not be empty"));
            }
            if text.len() > 65536 {
                return Err(Error::Input("text exceeds 64 KiB"));
            }
            bytes = bytes.saturating_add(text.len());
        }
        if bytes > 4 * 1024 * 1024 {
            return Err(Error::Input("texts exceed 4 MiB"));
        }
        texts
            .iter()
            .map(|text| {
                self.one(text.as_ref())
                    .map_err(|e| Error::Inference(e.to_string()))
            })
            .collect()
    }

    fn one(&self, text: &str) -> candle_core::Result<Vec<f32>> {
        let encoded = self
            .inner
            .tokenizer
            .encode(text, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let n = encoded.len();
        let ids = Tensor::new(encoded.get_ids(), &Device::Cpu)?.unsqueeze(0)?;
        let types = Tensor::new(encoded.get_type_ids(), &Device::Cpu)?.unsqueeze(0)?;
        let mask = Tensor::new(encoded.get_attention_mask(), &Device::Cpu)?.unsqueeze(0)?;
        let hidden = self.inner.model.forward(&ids, &types, Some(&mask))?;
        let weights = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
        let pooled = hidden
            .broadcast_mul(&weights)?
            .sum(1)?
            .squeeze(0)?
            .to_vec1::<f32>()?;
        let norm = pooled
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        if n == 0 || !norm.is_finite() || norm <= 0.0 {
            candle_core::bail!("non-finite or zero embedding");
        }
        Ok(pooled
            .into_iter()
            .map(|v| (v as f64 / norm) as f32)
            .collect())
    }
}

fn tokenizer() -> candle_core::Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_bytes(include_bytes!("../assets/tokenizer.json"))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 128,
            ..Default::default()
        }))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    Ok(tokenizer)
}

fn load_bundled() -> candle_core::Result<Inner> {
    let file = safetensors::SafeTensors::deserialize(include_bytes!("../assets/model.safetensors"))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let shapes: HashMap<String, Vec<usize>> =
        serde_json::from_str(include_str!("../assets/shapes.json"))
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let floats = |bytes: &[u8]| {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>()
    };
    let mut tensors = HashMap::new();
    for (name, shape) in shapes {
        let data = if shape.len() < 2 {
            floats(
                file.tensor(&name)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                    .data(),
            )
        } else {
            let packed = file
                .tensor(&format!("{name}.q4"))
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let scales = file
                .tensor(&format!("{name}.scales"))
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let scales = scales
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::f16::from_bits(u16::from_le_bytes(*b)).to_f32())
                .collect::<Vec<_>>();
            (0..shape.iter().product())
                .map(|i| {
                    let nibble = (packed.data()[i / 2] >> ((i % 2) * 4)) & 15;
                    (nibble as i16 - 8) as f32 * scales[i / 32]
                })
                .collect()
        };
        tensors.insert(name, Tensor::from_vec(data, shape, &Device::Cpu)?);
    }
    let config: Config = serde_json::from_str(include_str!("../assets/config.json"))
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let model = BertModel::load(
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
        &config,
    )?;
    Ok(Inner {
        model,
        tokenizer: tokenizer()?,
    })
}
