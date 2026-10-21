use crate::{Document, Error, Hit, Index, Result, validate_query};
use serde::Serialize;

/// Caller-owned, query-time documents. Implementations own discovery, I/O and
/// caching; the search library never guesses paths or writes a cache.
/// Return at most 64 candidates, not an entire unbounded log or database.
pub trait DynamicSource {
    /// Collect bounded candidates to embed for this query.
    fn documents(&self, query: &str) -> Result<Vec<Document>>;
}

impl<F> DynamicSource for F
where
    F: Fn(&str) -> Result<Vec<Document>>,
{
    fn documents(&self, query: &str) -> Result<Vec<Document>> {
        self(query)
    }
}

/// One supplemental source. Weight orders supplemental matches only: maintained
/// primary sources always precede dynamic material, regardless of similarity.
pub struct WeightedSource<'a> {
    /// A stable label for source failures.
    pub name: &'a str,
    /// Relative supplemental weight, finite and greater than zero, at most one.
    pub weight: f64,
    /// Caller-owned implementation.
    pub source: &'a dyn DynamicSource,
}

/// A supplemental failure does not suppress working primary search results.
#[derive(Debug, Serialize)]
pub struct SourceIssue {
    /// The caller's source label.
    pub source: String,
    /// Diagnostic text, not an execution instruction.
    pub error: String,
}

/// Maintained matches followed by weighted supplemental matches and diagnostics.
#[derive(Debug, Serialize)]
pub struct SearchResults {
    /// Primary matches first; historical/dynamic matches can only follow them.
    pub matches: Vec<Hit>,
    /// Sources that could not be searched.
    pub issues: Vec<SourceIssue>,
}

impl Index {
    /// Search compiled guidance, then embed caller-supplied query-time sources.
    /// Each limit is at most 20; at most eight dynamic sources are accepted.
    /// Source errors are reported without hiding maintained guidance.
    pub fn search_with_sources(
        &self,
        query: &str,
        primary_limit: usize,
        sources: &[WeightedSource<'_>],
        supplemental_limit: usize,
    ) -> Result<SearchResults> {
        validate_query(query)?;
        if supplemental_limit > 20 || sources.len() > 8 {
            return Err(Error::InvalidInput(
                "dynamic source or result limit exceeded",
            ));
        }
        if sources
            .iter()
            .any(|s| !s.weight.is_finite() || s.weight <= 0.0 || s.weight > 1.0)
        {
            return Err(Error::InvalidInput(
                "source weight must be finite and in (0, 1]",
            ));
        }
        let mut result = SearchResults {
            matches: self.search(query, primary_limit)?,
            issues: Vec::new(),
        };
        if supplemental_limit == 0 {
            return Ok(result);
        }
        let mut supplemental = Vec::new();
        for source in sources {
            let hits = (|| {
                let documents = source.source.documents(query)?;
                if documents.len() > 64 {
                    return Err(Error::InvalidInput("dynamic source exceeds 64 candidates"));
                }
                Index::new(documents)?.search(query, supplemental_limit)
            })();
            match hits {
                Ok(hits) => supplemental.extend(hits.into_iter().map(|mut hit| {
                    hit.score *= source.weight;
                    hit
                })),
                Err(error) => result.issues.push(SourceIssue {
                    source: source.name.into(),
                    error: error.to_string(),
                }),
            }
        }
        supplemental.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut groups = std::collections::BTreeSet::new();
        result.matches.extend(
            supplemental
                .into_iter()
                .filter(|hit| groups.insert(hit.document.group()))
                .take(supplemental_limit),
        );
        Ok(result)
    }
}

/// Cheap bounded candidate selection before embedding a large caller-owned
/// collection. Uses the same lexical retrieval as normal search, with no model
/// initialization, filesystem access, or requirement for prepared vectors.
pub fn lexical_candidates(
    documents: &[Document],
    query: &str,
    limit: usize,
) -> Result<Vec<Document>> {
    validate_query(query)?;
    if limit > 64 {
        return Err(Error::InvalidInput("candidate limit exceeds 64"));
    }
    crate::index::validate_documents(documents)?;
    Ok(crate::index::Lexical::new(documents)
        .search(query)
        .into_iter()
        .take(limit)
        .map(|(index, _)| documents[index].clone())
        .collect())
}
