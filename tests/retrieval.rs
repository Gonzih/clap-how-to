use clap_how_to::{Document, Index, Mode, PreparedIndex, Source};

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
