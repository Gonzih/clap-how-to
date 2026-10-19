use clap_how_to::{Document, Index, Mode, PreparedIndex, Source};

#[test]
fn after_help_adds_lexical_evidence_without_changing_neural_coordinates()
-> Result<(), Box<dyn std::error::Error>> {
    use clap::{Arg, Command};
    use clap_how_to::command_documents;

    let tree = |details: bool| {
        let mut app = Command::new("app").about("A collection of tools");
        let mut vectors = Command::new("embeddings").about("Draw vector relationships");
        let mut bare = Command::new("bare");
        let mut arguments = Command::new("arguments")
            .arg(Arg::new("width").long("width").help("Sliding window width"));
        let mut short = Command::new("short").about("Inspect a record");
        if details {
            app = app.after_long_help("Root-only operational details");
            vectors = vectors
                .after_help("Obsolete short-form details")
                .after_long_help(
                    "Edges use full vector distances, not PCA coordinates or projection distances.",
                );
            bare = bare.after_long_help("Details must not replace the canonical intent fallback");
            arguments = arguments.after_long_help("Details must not replace argument fallback");
            short = short.after_help("Short-form details are available too");
        }
        app.subcommand(vectors)
            .subcommand(bare)
            .subcommand(arguments)
            .subcommand(short)
    };
    let before = Index::new(command_documents(&tree(false)))?;
    let after = Index::new(command_documents(&tree(true)))?;
    assert_eq!(before.prepared().model, after.prepared().model);
    assert_eq!(before.prepared().embeddings, after.prepared().embeddings);
    assert!(
        before
            .search_with_mode("PCA coordinates projection", 5, Mode::Lexical)?
            .is_empty()
    );
    let hits = after.search_with_mode("PCA coordinates projection", 5, Mode::Lexical)?;
    assert_eq!(hits[0].document.title, "embeddings");
    assert!(!hits[0].document.text.contains("Obsolete short-form"));
    assert!(!hits[0].document.text.contains("Root-only"));
    assert!(
        after
            .prepared()
            .documents
            .iter()
            .find(|d| d.title == "short")
            .unwrap()
            .text
            .contains("Short-form details")
    );
    assert!(
        after
            .prepared()
            .documents
            .iter()
            .find(|d| d.title == "bare")
            .unwrap()
            .text
            .starts_with("app bare\n")
    );
    for query in [
        "PCA coordinates projection",
        "draw relationships",
        "sliding window",
    ] {
        let old = before.search_with_mode(query, 5, Mode::Semantic)?;
        let new = after.search_with_mode(query, 5, Mode::Semantic)?;
        assert_eq!(old.len(), new.len());
        for (a, b) in old.iter().zip(new.iter()) {
            assert_eq!(a.document.source, b.document.source);
            assert_eq!(a.score.to_bits(), b.score.to_bits());
            assert_eq!(
                a.semantic_score.map(f32::to_bits),
                b.semantic_score.map(f32::to_bits)
            );
        }
    }
    Ok(())
}

fn document(title: &str, text: &str) -> Document {
    Document {
        title: title.into(),
        text: text.into(),
        source: Source::Command {
            command: title.into(),
            help: format!("{title} --help"),
        },
    }
}

#[test]
fn semantic_no_overlap_roundtrip_and_coordinate_validation()
-> Result<(), Box<dyn std::error::Error>> {
    let documents = vec![
        document("backup", "Save a recoverable snapshot of all project data."),
        document(
            "delete",
            "Permanently remove a snapshot and its stored contents.",
        ),
        document(
            "login",
            "Authenticate the current user using browser OAuth authorization.",
        ),
        document(
            "pay",
            "Transfer money to a recipient using the configured wallet.",
        ),
    ];
    let index = Index::new(documents)?;
    let query = "protect me from losing everything";
    assert!(index.search_with_mode(query, 5, Mode::Lexical)?.is_empty());
    assert_eq!(
        index.search_with_mode(query, 5, Mode::Semantic)?[0]
            .document
            .title,
        "backup"
    );
    let json = serde_json::to_string(index.prepared())?;
    let restored = Index::from_prepared(serde_json::from_str(&json)?)?;
    assert_eq!(restored.search(query, 1)?[0].document.title, "backup");
    assert!(restored.search("hello", 21).is_err());
    assert!(restored.search("\u{1b}", 5).is_err());
    let mut prepared: PreparedIndex = serde_json::from_str(&json)?;
    prepared.model.revision.push('x');
    assert!(Index::from_prepared(prepared).is_err());
    let mut prepared: PreparedIndex = serde_json::from_str(&json)?;
    prepared.embeddings[0][0] = f32::NAN;
    assert!(Index::from_prepared(prepared).is_err());
    let mut prepared: PreparedIndex = serde_json::from_str(&json)?;
    prepared.embeddings[0].pop();
    assert!(Index::from_prepared(prepared).is_err());
    let mut prepared: PreparedIndex = serde_json::from_str(&json)?;
    prepared.embeddings.pop();
    assert!(Index::from_prepared(prepared).is_err());
    Ok(())
}
