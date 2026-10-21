use clap_how_to::{Source, markdown_documents};

#[test]
fn yaml_is_not_guidance_and_introductions_keep_their_examples() {
    let source = "---\nname: private-metadata\n---\n# Messaging\n\nRead pending messages with:\n\n```bash\napp inbox --peek\n```\n\n# Other\n\n**Orphan heading:**\n\n# Final\n\nStandalone guidance.\n";
    let docs = markdown_documents("skill.md", source, None);
    assert_eq!(docs.len(), 2);
    assert_eq!(
        docs[0].text,
        "Read pending messages with:\n\n```bash\napp inbox --peek\n```"
    );
    assert!(matches!(
        docs[0].source,
        Source::Documentation { line: 6, .. }
    ));
    assert!(docs.iter().all(|d| !d.text.contains("private-metadata")));
}

#[test]
fn merging_does_not_reintroduce_excluded_query_examples() {
    let source =
        "# Help\n\nSearch with:\n\napp how \"find a message\"\n\n```bash\napp inbox\n```\n";
    let docs = markdown_documents("guide.md", source, None);
    assert!(
        docs.iter()
            .all(|d| !d.text.contains("app how") && !d.text.contains("Search with:"))
    );
    assert!(docs.iter().any(|d| d.text.contains("app inbox")));
}
