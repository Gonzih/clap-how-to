//! Compare bundled vectors with upstream F32 weights using the same Rust runtime.
//! Explicit developer tool; requires original weights and a JSON array of texts.
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config};
    use tokenizers::{Tokenizer, TruncationParams};
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let root = std::path::Path::new(&args[0]);
    let texts: Vec<String> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let tensors = candle_core::safetensors::load(root.join("model.safetensors"), &Device::Cpu)?;
    let config: Config = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
    let model = BertModel::load(
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
        &config,
    )?;
    let mut tokenizer =
        Tokenizer::from_file(root.join("tokenizer.json")).map_err(|e| e.to_string())?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 128,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for text in &texts {
        let tokens = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| e.to_string())?;
        let ids = Tensor::new(tokens.get_ids(), &Device::Cpu)?.unsqueeze(0)?;
        let types = ids.zeros_like()?;
        let output = model.forward(&ids, &types, None)?;
        let mut pooled = output.sum(1)?.squeeze(0)?.to_vec1::<f32>()?;
        let norm = pooled
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        for v in &mut pooled {
            *v = (*v as f64 / norm) as f32;
        }
        rows.push(pooled);
    }
    println!("{}", serde_json::json!({"embeddings":rows}));
    Ok(())
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("The upstream-weight reference exporter requires Candle on a non-Linux-ARM64 host. Bundled Encoder inference is supported on Linux ARM64.".into())
}
