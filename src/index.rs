use crate::{Document, Encoder, Error, ModelInfo, Result, model_info, validate_query};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Optional retrieval ablation for evaluation. Hybrid is the normal search mode.
#[derive(Clone, Copy, Debug, Default)]
pub enum Mode {
    /// Neural cosine similarity fused with lexical BM25 by reciprocal rank.
    #[default]
    Hybrid,
    /// Neural cosine only, without keyword rules or synonym dictionaries.
    Semantic,
    /// BM25 only, useful for exact command and flag names.
    Lexical,
}

/// Portable source documents and their coordinates. Generate during development,
/// serialize with serde_json and compile the JSON into a CLI with include_str!.
/// Queries are never part of this artifact. Treat external artifacts as untrusted
/// until validated by [`Index::from_prepared`]; provenance is caller-owned text.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedIndex {
    /// Exact encoder coordinate-space contract.
    pub model: ModelInfo,
    /// Source text and provenance retained without generated instructions.
    pub documents: Vec<Document>,
    /// Unit-length vectors in document order.
    pub embeddings: Vec<Vec<f32>>,
}

/// A retrieved source. Scores are rankings, never confidence or execution authority.
#[derive(Clone, Debug, Serialize)]
pub struct Hit {
    /// Original source document.
    #[serde(flatten)]
    pub document: Document,
    /// RRF score in hybrid mode, cosine or BM25 in ablation modes.
    pub score: f64,
    /// Neural cosine when evaluated; absent in lexical-only mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_score: Option<f32>,
}

/// Exact vector scan plus a small in-memory BM25 index. No vector database,
/// filesystem, network, query logging, command execution or generated answers.
pub struct Index {
    prepared: PreparedIndex,
    lexical: Lexical,
}

impl Index {
    /// Encode a corpus once. For short-lived CLIs, prefer shipping a prepared index.
    pub fn new(documents: Vec<Document>) -> Result<Self> {
        validate_documents(&documents)?;
        let texts = documents
            .iter()
            .map(Document::semantic_text)
            .collect::<Vec<_>>();
        let embeddings = if texts.is_empty() {
            Vec::new()
        } else {
            Encoder::new()
                .map_err(|e| Error::Encoder(e.to_string()))?
                .embed(&texts)
                .map_err(|e| Error::Encoder(e.to_string()))?
        };
        Self::from_prepared(PreparedIndex {
            model: model_info(),
            documents,
            embeddings,
        })
    }

    /// Validate model identity, count, dimensions, unit norms and finite values.
    /// This does not authenticate caller-supplied document contents or URLs.
    pub fn from_prepared(prepared: PreparedIndex) -> Result<Self> {
        validate_documents(&prepared.documents)?;
        if prepared.model != model_info() {
            return Err(Error::InvalidInput("index encoder fingerprint mismatch"));
        }
        if prepared.documents.len() != prepared.embeddings.len() {
            return Err(Error::InvalidInput("index row count mismatch"));
        }
        for row in &prepared.embeddings {
            let norm = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
            if row.len() != 384 || !row.iter().all(|v| v.is_finite()) || (norm - 1.0).abs() > 0.002
            {
                return Err(Error::InvalidInput(
                    "index vector must be finite, normalized and 384-dimensional",
                ));
            }
        }
        let lexical = Lexical::new(&prepared.documents);
        Ok(Self { prepared, lexical })
    }

    /// Borrow the serializable prepared corpus. Contains no query data.
    pub fn prepared(&self) -> &PreparedIndex {
        &self.prepared
    }

    /// Return up to 20 matches using hybrid retrieval. Empty/no lexical evidence
    /// does not prove no answer exists: neural retrieval is approximate in meaning.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        self.search_with_mode(query, limit, Mode::Hybrid)
    }

    /// Evaluate one retrieval mode with the same corpus and stable tie ordering.
    pub fn search_with_mode(&self, query: &str, limit: usize, mode: Mode) -> Result<Vec<Hit>> {
        validate_query(query)?;
        if limit > 20 {
            return Err(Error::InvalidInput("result limit exceeds 20"));
        }
        if limit == 0 || self.prepared.documents.is_empty() {
            return Ok(Vec::new());
        }
        let n = self.prepared.documents.len();
        let lexical = self.lexical.search(query);
        let mut cosine = vec![None; n];
        let semantic = if matches!(mode, Mode::Lexical) {
            Vec::new()
        } else {
            let vector = Encoder::new()
                .map_err(|e| Error::Encoder(e.to_string()))?
                .embed(&[query])
                .map_err(|e| Error::Encoder(e.to_string()))?
                .remove(0);
            let mut scores = self
                .prepared
                .embeddings
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    let score = row
                        .iter()
                        .zip(&vector)
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        .clamp(-1.0, 1.0);
                    cosine[i] = Some(score);
                    (i, score as f64)
                })
                .collect::<Vec<_>>();
            order(&mut scores);
            scores
        };
        let mut ranked = match mode {
            Mode::Semantic => semantic,
            Mode::Lexical => lexical,
            Mode::Hybrid => {
                let mut scores = vec![0.0; n];
                for (ranking, weight) in [(&semantic, 2.0), (&lexical, 1.0)] {
                    for (rank, (i, _)) in ranking.iter().take(100).enumerate() {
                        scores[*i] += weight / (10 + rank + 1) as f64;
                    }
                }
                scores
                    .into_iter()
                    .enumerate()
                    .filter(|(_, score)| *score > 0.0)
                    .collect()
            }
        };
        order(&mut ranked);
        let mut groups = BTreeSet::new();
        Ok(ranked
            .into_iter()
            .filter(|(i, _)| groups.insert(self.prepared.documents[*i].group()))
            .take(limit)
            .map(|(i, score)| Hit {
                document: self.prepared.documents[i].clone(),
                score,
                semantic_score: cosine[i],
            })
            .collect())
    }
}

fn validate_documents(documents: &[Document]) -> Result<()> {
    if documents.len() > 4096 {
        return Err(Error::InvalidInput("index exceeds 4096 documents"));
    }
    let mut total = 0;
    for document in documents {
        if document.title.len() > 512
            || document.text.len() > 16384
            || document.text.trim().is_empty()
        {
            return Err(Error::InvalidInput(
                "document title/text invalid or too large",
            ));
        }
        total += document.title.len() * 2 + document.text.len() + 2;
    }
    if total > 4 * 1024 * 1024 {
        return Err(Error::InvalidInput("index text exceeds 4 MiB"));
    }
    Ok(())
}

fn order(scores: &mut [(usize, f64)]) {
    scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
}

// Standard BM25 and reciprocal-rank fusion, following the small-corpus approach
// used by Gonzih/ontology (MIT). No hardcoded product vocabulary or synonyms.
fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .filter(|s| {
            !matches!(
                s.as_str(),
                "a" | "an"
                    | "the"
                    | "and"
                    | "or"
                    | "is"
                    | "are"
                    | "was"
                    | "were"
                    | "be"
                    | "been"
                    | "to"
                    | "of"
                    | "for"
                    | "from"
                    | "with"
                    | "by"
                    | "in"
                    | "on"
                    | "at"
                    | "this"
                    | "that"
                    | "it"
                    | "its"
                    | "my"
                    | "me"
                    | "i"
                    | "you"
                    | "your"
                    | "their"
                    | "our"
                    | "which"
                    | "what"
                    | "when"
                    | "where"
                    | "how"
                    | "why"
                    | "can"
                    | "could"
                    | "do"
                    | "does"
                    | "did"
                    | "have"
                    | "has"
                    | "had"
                    | "so"
            )
        })
        .collect()
}
struct Lexical {
    lengths: Vec<usize>,
    postings: BTreeMap<String, Vec<(usize, usize)>>,
}
impl Lexical {
    fn new(documents: &[Document]) -> Self {
        let mut postings = BTreeMap::<String, Vec<(usize, usize)>>::new();
        let mut lengths = Vec::new();
        for (i, document) in documents.iter().enumerate() {
            let terms = tokens(&document.indexed_text());
            lengths.push(terms.len());
            let mut counts = BTreeMap::new();
            for term in terms {
                *counts.entry(term).or_insert(0) += 1;
            }
            for (term, count) in counts {
                postings.entry(term).or_default().push((i, count));
            }
        }
        Self { lengths, postings }
    }
    fn search(&self, query: &str) -> Vec<(usize, f64)> {
        let count = self.lengths.len() as f64;
        if count == 0.0 {
            return Vec::new();
        }
        let average = self.lengths.iter().sum::<usize>() as f64 / count;
        let mut scores = vec![0.0; self.lengths.len()];
        for term in tokens(query).into_iter().collect::<BTreeSet<_>>() {
            if let Some(rows) = self.postings.get(&term) {
                let df = rows.len() as f64;
                let idf = (1.0 + (count - df + 0.5) / (df + 0.5)).ln();
                for &(i, tf) in rows {
                    let tf = tf as f64;
                    scores[i] += idf * tf * 2.2
                        / (tf + 1.2 * (0.25 + 0.75 * self.lengths[i] as f64 / average.max(1.0)));
                }
            }
        }
        let mut scores = scores
            .into_iter()
            .enumerate()
            .filter(|(_, s)| *s > 0.0)
            .collect::<Vec<_>>();
        order(&mut scores);
        scores
    }
}
