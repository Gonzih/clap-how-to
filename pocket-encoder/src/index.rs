use crate::{Encoder, Error, ModelInfo, model_info};

const DIMENSIONS: usize = 384;
const MAX_VECTORS: usize = 4096;
const MAGIC: &[u8; 8] = b"PKTIDX01";
const HEADER_BYTES: usize = 16;
const MAX_MODEL_BYTES: usize = 4096;

/// One exact cosine match. Row IDs follow the original embedding order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    /// Zero-based row in the indexed corpus; labels remain caller-owned.
    pub index: usize,
    /// Cosine similarity, clamped to [-1, 1]; not a confidence probability.
    pub score: f32,
}

/// Immutable exact search over at most 4096 normalized, bundled-model vectors.
///
/// Stores coordinates in one contiguous allocation. Texts and queries are not
/// retained. Loading and vector search do not initialize the neural encoder.
#[derive(Clone, Debug)]
pub struct Index {
    model: ModelInfo,
    vectors: Vec<f32>,
}

impl Index {
    /// Encode a corpus once. For short-lived processes, ship [`Self::to_bytes`]
    /// and restore it with [`Self::from_bytes`] instead of encoding on startup.
    pub fn new(texts: &[impl AsRef<str>]) -> Result<Self, Error> {
        if texts.len() > MAX_VECTORS {
            return Err(Error::Input("index exceeds 4096 vectors"));
        }
        let embeddings = if texts.is_empty() {
            Vec::new()
        } else {
            Encoder::new()?.embed(texts)?
        };
        Self::from_embeddings(model_info(), embeddings)
    }

    /// Validate the complete model identity, row count, dimensions, finite
    /// values and unit norms, then pack rows in their original order.
    pub fn from_embeddings(model: ModelInfo, embeddings: Vec<Vec<f32>>) -> Result<Self, Error> {
        validate_model(&model)?;
        if embeddings.len() > MAX_VECTORS {
            return Err(Error::Input("index exceeds 4096 vectors"));
        }
        let mut vectors = Vec::with_capacity(embeddings.len() * DIMENSIONS);
        for row in embeddings {
            validate_vector(&row)?;
            vectors.extend(row);
        }
        Ok(Self { model, vectors })
    }

    /// Coordinate-space identity, available without initializing the encoder.
    pub fn model(&self) -> &ModelInfo {
        &self.model
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.vectors.len() / DIMENSIONS
    }

    /// Whether the corpus has no vectors.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Encode a question and return up to `limit` exact matches, best first.
    /// Equal scores favor earlier rows. A zero limit or empty corpus avoids
    /// inference, but still rejects blank or oversized questions.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Neighbor>, Error> {
        if query.trim().is_empty() {
            return Err(Error::Input("text must not be empty"));
        }
        if query.len() > 65536 {
            return Err(Error::Input("text exceeds 64 KiB"));
        }
        if limit == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let vector = Encoder::new()?.embed(&[query])?.remove(0);
        Ok(self.nearest(&vector, limit))
    }

    /// Search with an already encoded question after validating its complete
    /// model identity, dimensions, finite values and unit norm.
    pub fn search_embedding(
        &self,
        model: &ModelInfo,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<Neighbor>, Error> {
        if model != &self.model {
            return Err(Error::Input("index encoder fingerprint mismatch"));
        }
        validate_vector(query)?;
        Ok(self.nearest(query, limit))
    }

    fn nearest(&self, query: &[f32], limit: usize) -> Vec<Neighbor> {
        let limit = limit.min(self.len());
        if limit == 0 {
            return Vec::new();
        }
        let mut hits = self
            .vectors
            .as_chunks::<DIMENSIONS>()
            .0
            .iter()
            .enumerate()
            .map(|(index, row)| Neighbor {
                index,
                score: row
                    .iter()
                    .zip(query)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    .clamp(-1.0, 1.0),
            })
            .collect::<Vec<_>>();
        let order =
            |a: &Neighbor, b: &Neighbor| b.score.total_cmp(&a.score).then(a.index.cmp(&b.index));
        if limit < hits.len() {
            hits.select_nth_unstable_by(limit, order);
            hits.truncate(limit);
        }
        hits.sort_unstable_by(order);
        hits
    }

    /// Serialize versioned metadata and exact little-endian F32 coordinates.
    /// Labels and queries are not included; no filesystem I/O is performed.
    pub fn to_bytes(&self) -> Vec<u8> {
        let model = serde_json::to_vec(&self.model).expect("serializable model metadata");
        let mut bytes = Vec::with_capacity(HEADER_BYTES + model.len() + self.vectors.len() * 4);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(model.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.len() as u32).to_le_bytes());
        bytes.extend(model);
        for value in &self.vectors {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// Load a binary index without neural initialization. Check the version,
    /// exact byte length, bounded counts, model fingerprint and every vector.
    /// This validates coordinates, not the authenticity of caller-owned data.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() < HEADER_BYTES || &bytes[..8] != MAGIC {
            return Err(Error::Input("invalid index header or version"));
        }
        let model_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        if model_len > MAX_MODEL_BYTES || count > MAX_VECTORS {
            return Err(Error::Input("index metadata or row count exceeds bounds"));
        }
        // The bounded header values make these additions/multiplications safe,
        // including on 32-bit targets. Reject truncation and trailing bytes.
        let start = HEADER_BYTES + model_len;
        if bytes.len() != start + count * DIMENSIONS * 4 {
            return Err(Error::Input("index byte length mismatch"));
        }
        let model = serde_json::from_slice(&bytes[HEADER_BYTES..start])
            .map_err(|_| Error::Input("invalid index model metadata"))?;
        validate_model(&model)?;
        let vectors = bytes[start..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .collect::<Vec<_>>();
        for row in vectors.as_chunks::<DIMENSIONS>().0 {
            validate_vector(row)?;
        }
        Ok(Self { model, vectors })
    }
}

fn validate_model(model: &ModelInfo) -> Result<(), Error> {
    if model != &model_info() {
        return Err(Error::Input("index encoder fingerprint mismatch"));
    }
    Ok(())
}

fn validate_vector(vector: &[f32]) -> Result<(), Error> {
    let valid = vector.len() == DIMENSIONS && {
        // A finite sum of squares also rejects every NaN/infinite coordinate.
        let norm = vector.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        norm.is_finite() && (norm - 1.0).abs() <= 0.002
    };
    if !valid {
        return Err(Error::Input(
            "index vector must be finite, normalized and 384-dimensional",
        ));
    }
    Ok(())
}
