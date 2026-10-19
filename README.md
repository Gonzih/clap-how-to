# clap-how-to

Offline semantic help search for Clap. Add `app how "task"` and
`app how-to "question"` to the same command tree. Retrieve real commands and
bundled documentation by meaning. Nothing executes a match or invents an answer.

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

For short-lived CLIs, encode the corpus during development. Serialize
`index.prepared()` and ship that JSON with `include_str!`; validate it with
`Index::from_prepared`. Only the question needs inference at runtime. Rebuild
the artifact when help/docs or the encoder fingerprint changes. No build script,
model cache, network service, Python or ONNX shared library is used at runtime.

The encoder runs all three MiniLM attention layers with Candle on CPU. Bundled
matrices use 4-bit storage; they expand to F32 once when loaded. This is actual
contextual neural inference, not feature hashing or a static word-vector lookup.
The small English model and 128-token truncation have limits: retrieve sources,
read their help, and apply your own judgment. Scores are rankings, not confidence.

See [encoder details](pocket-encoder/README.md), [design and alternatives](docs/design.md),
[evaluation](benchmarks/README.md) and the runnable `examples/assistant.rs`.
Source is MIT; bundled model/tokenizer are Apache-2.0, with attribution in
[`pocket-encoder/NOTICE`](pocket-encoder/NOTICE).
