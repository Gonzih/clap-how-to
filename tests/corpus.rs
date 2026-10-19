use clap::{Arg, Command};
use clap_how_to::{HowToExt, Source, command_documents, markdown_documents, validate_query};

#[test]
fn actual_tree_aliases_hidden_items_and_literal_question() {
    let bare = command_documents(&Command::new("bare").subcommand(Command::new("list")));
    assert_eq!(bare.len(), 2);
    assert!(bare.iter().all(|d| !d.text.is_empty()));
    let app = Command::new("app")
        .arg(
            Arg::new("token")
                .long("token")
                .global(true)
                .help("secret credentials"),
        )
        .subcommand(
            Command::new("send")
                .visible_alias("deliver")
                .about("Transfer text")
                .arg(Arg::new("text").help("Exact message body")),
        )
        .subcommand(Command::new("secret").hide(true).about("hidden text"))
        .with_how_to();
    let docs = command_documents(&app);
    assert_eq!(docs.len(), 2);
    assert!(docs[0].text.contains("secret credentials"));
    assert!(docs[1].text.contains("Exact message body"));
    assert!(!docs[1].text.contains("secret credentials"));
    assert!(docs[1].text.contains("deliver"));
    match &docs[1].source {
        Source::Command { command, help } => {
            assert_eq!(command, "app send");
            assert_eq!(help, "app send --help");
        }
        _ => panic!("command source"),
    }
    for name in ["how", "how-to"] {
        let matches = app
            .clone()
            .try_get_matches_from(["app", name, "--literal=quoted", "--"])
            .err();
        assert!(matches.is_some()); // flag-shaped text is not silently rewritten
        let matches = app
            .clone()
            .try_get_matches_from(["app", name, "a question"])
            .unwrap();
        let (_, args) = matches.subcommand().unwrap();
        assert_eq!(args.get_one::<String>("query").unwrap(), "a question");
    }
    assert!(validate_query(" ").is_err());
    assert!(validate_query("bad\u{1b}text").is_err());
    assert!(validate_query(&"x".repeat(513)).is_err());
}

#[test]
fn markdown_excerpts_are_exact_bounded_utf8_with_real_lines() {
    let source = format!(
        "# One\nFirst paragraph.\n\n```rust\n# code, not a heading\n```\n\n# Two\n{}\n",
        "é 漢字 ".repeat(300)
    );
    let docs = markdown_documents("guide.md", &source, Some("https://example.test/guide"));
    assert!(docs.len() > 4);
    for doc in docs {
        assert!(doc.text.len() <= 700 && source.contains(&doc.text));
        let Source::Documentation { line, section, .. } = doc.source else {
            panic!()
        };
        assert!(
            source
                .lines()
                .nth(line - 1)
                .unwrap()
                .contains(doc.text.lines().next().unwrap())
        );
        assert_ne!(section, "code, not a heading");
    }
}
