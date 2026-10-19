//! Offline semantic discovery for a real Clap command tree and bundled documents.
//!
//! The learned encoder, tokenizer and model metadata are compiled into the binary.
//! This crate never downloads a model, opens a cache, parses an action or executes
//! a result. Applications retain their own output, authorization and focus policy.

mod corpus;
mod index;

use clap::{Args, Command};
pub use corpus::{Document, Source, command_documents, markdown_documents};
pub use index::{Hit, Index, Mode, PreparedIndex};
pub use pocket_encoder::{Encoder, ModelInfo, model_info};
use serde::Serialize;

/// Errors are returned to the caller; this library does not print or exit.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The query is empty, too long, or contains controls.
    #[error("{0}")]
    InvalidQuery(&'static str),
    /// A caller-supplied corpus or encoder batch exceeds its documented bounds.
    #[error("{0}")]
    InvalidInput(&'static str),
    /// The compiled model could not be initialized.
    #[error("bundled semantic encoder: {0}")]
    Encoder(String),
}

/// Result type used by the library.
pub type Result<T> = std::result::Result<T, Error>;

/// Query arguments usable in a derived `how-to` variant (with visible alias `how`).
#[derive(Args, Debug, Clone, Serialize)]
pub struct HowToArgs {
    /// Describe a task or ask a question; results are source matches, not actions.
    pub query: String,
}

/// Build a `how-to QUERY` subcommand with the equivalent visible alias `how`.
pub fn command() -> Command {
    HowToArgs::augment_args(
        Command::new("how-to")
            .visible_alias("how")
            .about("Find relevant commands and bundled guidance by meaning, offline"),
    )
}

/// Builder integration: `Command::new("app").with_how_to()`.
pub trait HowToExt: Sized {
    /// Add the two equivalent discovery spellings without an execution handler.
    fn with_how_to(self) -> Self;
}

impl HowToExt for Command {
    fn with_how_to(self) -> Self {
        self.subcommand(command())
    }
}

/// Validate a CLI question without loading the encoder or touching the filesystem.
pub fn validate_query(query: &str) -> Result<()> {
    if query.trim().is_empty() {
        return Err(Error::InvalidQuery("question must not be empty"));
    }
    if query.len() > 512 {
        return Err(Error::InvalidQuery("question exceeds 512 bytes"));
    }
    if query.chars().any(char::is_control) {
        return Err(Error::InvalidQuery("question must be control-free"));
    }
    Ok(())
}
