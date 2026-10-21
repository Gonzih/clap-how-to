use clap_how_to::{
    Document, DynamicSource, Error, Index, Source, WeightedSource, lexical_candidates,
};

struct Records(Vec<Document>);
impl DynamicSource for Records {
    fn documents(&self, _: &str) -> clap_how_to::Result<Vec<Document>> {
        Ok(self.0.clone())
    }
}
fn record(provider: &str, text: &str) -> Document {
    Document {
        title: "historical record".into(),
        text: text.into(),
        source: Source::Record {
            provider: provider.into(),
            source: format!("{provider}.jsonl"),
            byte_offset: 42,
        },
    }
}
fn primary() -> Index {
    Index::new(vec![Document {
        title: "status".into(),
        text: "List active jobs".into(),
        source: Source::Command {
            command: "app status".into(),
            help: "app status --help".into(),
        },
    }])
    .unwrap()
}

#[test]
fn maintained_guidance_precedes_even_a_better_matching_historical_record() {
    let source = Records(vec![record(
        "codex",
        "Reply to a channel message with app reply",
    )]);
    let result = primary()
        .search_with_sources(
            "reply to a channel message",
            1,
            &[WeightedSource {
                name: "logs",
                weight: 0.05,
                source: &source,
            }],
            2,
        )
        .unwrap();
    assert_eq!(result.matches.len(), 2);
    assert!(matches!(
        result.matches[0].document.source,
        Source::Command { .. }
    ));
    assert!(matches!(
        result.matches[1].document.source,
        Source::Record {
            byte_offset: 42,
            ..
        }
    ));
    assert!(result.issues.is_empty());
}

#[test]
fn weights_order_supplemental_sources_and_logs_only_omits_primary_results() {
    let low = Records(vec![record("low", "Read message history")]);
    let high = Records(vec![record("high", "Read message history")]);
    let result = primary()
        .search_with_sources(
            "message history",
            0,
            &[
                WeightedSource {
                    name: "low",
                    weight: 0.01,
                    source: &low,
                },
                WeightedSource {
                    name: "high",
                    weight: 0.5,
                    source: &high,
                },
            ],
            2,
        )
        .unwrap();
    assert!(
        matches!(&result.matches[0].document.source,Source::Record{provider,..} if provider=="high")
    );
    assert!(
        result
            .matches
            .iter()
            .all(|hit| matches!(hit.document.source, Source::Record { .. }))
    );
}

#[test]
fn unavailable_and_oversized_sources_leave_primary_search_usable() {
    struct Broken;
    impl DynamicSource for Broken {
        fn documents(&self, _: &str) -> clap_how_to::Result<Vec<Document>> {
            Err(Error::Source("unavailable".into()))
        }
    }
    let large = Records(vec![record("large", "candidate"); 65]);
    let result = primary()
        .search_with_sources(
            "active jobs",
            1,
            &[
                WeightedSource {
                    name: "missing",
                    weight: 0.1,
                    source: &Broken,
                },
                WeightedSource {
                    name: "large",
                    weight: 0.1,
                    source: &large,
                },
            ],
            2,
        )
        .unwrap();
    assert_eq!(result.matches.len(), 1);
    assert_eq!(result.issues.len(), 2);
    assert!(result.issues[0].error.contains("unavailable"));
    assert!(result.issues[1].error.contains("64 candidates"));
}

#[test]
fn invalid_weights_are_rejected_and_lexical_candidates_do_not_need_vectors() {
    let source = Records(vec![]);
    for weight in [f64::NAN, 0.0, -1.0, 2.0] {
        assert!(
            primary()
                .search_with_sources(
                    "status",
                    1,
                    &[WeightedSource {
                        name: "bad",
                        weight,
                        source: &source
                    }],
                    1
                )
                .is_err()
        );
    }
    let documents = vec![
        record("one", "Run the canary migration"),
        record("two", "Reply to a channel"),
    ];
    assert_eq!(
        lexical_candidates(&documents, "canary migration", 1).unwrap(),
        vec![documents[0].clone()]
    );
    assert!(
        lexical_candidates(&documents, "unrelatedword", 5)
            .unwrap()
            .is_empty()
    );
}
