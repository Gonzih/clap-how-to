# clap-how-to

Offline semantic help search for Clap. Add `app how "task"` and
`app how-to "question"` to the same command tree. Retrieve real commands and
bundled documentation by meaning. Nothing executes a match or invents an answer.

```sh
cargo add clap-how-to
# For embeddings without Clap:
cargo add pocket-encoder
```

Two crates, one small interface each:

| Crate | Job |
| --- | --- |
| `clap-how-to` | Clap introspection, Markdown excerpts, neural + BM25 retrieval |
| `pocket-encoder` | Bundled 384-dimensional sentence transformer, native Rust CPU inference |

```rust,no_run
use clap::Command;
use clap_how_to::{HowToExt, Index, command_documents, markdown_documents};

let app = Command::new("example")
    .subcommand(Command::new("backup").about("Save a recoverable copy of your data"))
    .with_how_to();
let mut documents = command_documents(&app);
documents.extend(markdown_documents("guide.md", "# Backups\nKeep a recoverable copy.", None));
let index = Index::new(documents)?;
let hits = index.search("protect me from losing everything", 5)?;
println!("{}", serde_json::to_string(&hits)?);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`with_how_to()` only adds parsing. Applications call `Index::search` in their
own handler and keep their own display, authorization, focus and lifecycle rules.
For Clap derive, add a `HowTo(HowToArgs)` variant with
`#[command(name = "how-to", visible_alias = "how")]`.

Command search includes local option help and `after_long_help` (or `after_help`).
These details remain lexical evidence; neural command vectors use the existing
first intent line. Rebuild prepared corpora after help changes. Model fingerprints
validate coordinate compatibility, not whether source text is current.

For short-lived CLIs, encode the corpus during development. Serialize
`index.prepared()` and ship that JSON with `include_str!`; validate it with
`Index::from_prepared`. Only the question needs inference at runtime. Rebuild
the artifact when help/docs or the encoder fingerprint changes. No build script,
model cache, network service, Python or ONNX shared library is used at runtime.

The encoder runs all three MiniLM attention layers on CPU. Linux ARM64 uses
an F32-only Rust backend; other targets use Candle. No special compiler flags
are required. Bundled
matrices use 4-bit storage; they expand to F32 once when loaded. This is actual
contextual neural inference, not feature hashing or a static word-vector lookup.
The small English model and 128-token truncation have limits: retrieve sources,
read their help, and apply your own judgment. Scores are rankings, not confidence.

See [encoder details](https://github.com/Gonzih/clap-how-to/blob/main/pocket-encoder/README.md), [design and alternatives](docs/design.md),
[evaluation](benchmarks/README.md) and the runnable `examples/assistant.rs`.
Source is MIT; bundled model/tokenizer are Apache-2.0, with attribution in
[`pocket-encoder/NOTICE`](pocket-encoder/NOTICE).

## Dynamic sources

`Index::search_with_sources` accepts caller-owned `DynamicSource` implementations
(or closures). Primary help and documentation always come first. Supplemental
matches are embedded locally, ranked by their relative `WeightedSource::weight`,
and appended with their original provenance. A failed source is reported in
`SearchResults::issues` without suppressing working help results.

Sources own their filesystem/database access and cache policy. Return at most
64 candidates per source; `lexical_candidates` provides cheap candidate selection
before neural embedding. Source text is evidence, never execution authority.
`Source::Record` identifies historical records by provider, locator, and byte offset.

```rust,no_run
use clap_how_to::{Document, Index, Source, WeightedSource, lexical_candidates};
# fn main() -> clap_how_to::Result<()> {
let help = Index::new(vec![])?; // normally load your compiled PreparedIndex
let records = vec![Document {
    title: "Codex log".into(),
    text: "The last migration was checked with cargo test".into(),
    source: Source::Record {
        provider: "codex".into(), source: "session.jsonl".into(), byte_offset: 42,
    },
}];
let logs = |query: &str| lexical_candidates(&records, query, 6);
let result = help.search_with_sources("check the migration", 5, &[
    WeightedSource { name: "logs", weight: 0.05, source: &logs },
], 3)?;
assert!(result.issues.is_empty());
# Ok(()) }
```

The 0.2 API adds `Source::Record` and `Error::Source`. Update exhaustive matches
when upgrading from 0.1. No harness discovery or cache writes happen unless the
application supplies such a source.

Release 0.2.1 includes dynamic sources and pins pocket-encoder 0.1.2 with the
generic Linux ARM64 portability fix. The model fingerprint and PKTIDX01 storage
format are unchanged.
