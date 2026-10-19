# Design

`how-to` is a retrieval command. A question becomes one embedding; an exact
cosine scan ranks the small bundled corpus. BM25 supplies exact-name evidence;
reciprocal-rank fusion combines the two rankings. The application prints real
source matches. No generation, tools, shell, URLs, network or implicit action.

Clap remains the source of truth for command names, aliases, argument help and
possible values. Neural command retrieval uses the intent description, while BM25
also indexes full argument text. Generated help/version arguments are omitted. Documentation is caller-provided Markdown with exact excerpt,
heading, filename, line and optional URL provenance. Indexes are ordinary serde
data, not a database. One best excerpt per section avoids repetitive results.
Questions are capped at 512 UTF-8 bytes; result limits at 20; corpora at 4096
documents and 4 MiB of indexed text. The application owns outer JSON size limits
when loading an external prepared index. Bundled indexes need no deserialization
from untrusted network sources.

## Why two crates

`pocket-encoder` knows nothing about Clap, ICQ, ontology, file stores or vector
databases. It exposes an encoder and a serializable coordinate-space identity.
That identity travels with ontology/datapack vectors, prepared CLI indexes and
query encoders. An index created with a different model/tokenizer/preprocessing
contract must be rebuilt, not silently compared because dimensions happen to fit.

`clap-how-to` adds a Clap adapter, source extraction and small-corpus retrieval.
It neither prints/exits nor decides whether a harness is muted. Consumers run
their own existing authorization/focus checks before calling it. Search text,
documentation and model matches never become execution permission.

## Alternatives considered

| Approach | Reason for the choice |
| --- | --- |
| Clap suggestions | Jaro string similarity repairs names, not task meaning. Kept separate. |
| Feature hashing | Fast lexical baseline used by ontology; no learned synonym meaning. |
| BM25 / Tantivy | Excellent lexical retrieval. A small in-memory BM25 index is enough here. |
| Nucleo | Fuzzy interactive name matching, not sentence meaning. |
| Model2Vec | Fast learned static embeddings. Useful alternative, but this task requested actual contextual neural inference. |
| FastEmbed / ONNX Runtime | Established Rust embedding API, but native Rust inference and compiled-in assets were preferred over a runtime/model download path. |
| RTen / Tract | Native Rust ONNX runtimes. Viable alternatives; not needed for this fixed three-layer BERT architecture. |
| Candle + MiniLM-L3 | Existing Rust BERT implementation, small sentence-trained model, fast measured CPU queries, no separate inference runtime. Chosen. |

We do not claim a new embedding architecture or a universally best model.
The included English smoke benchmark is inspectable, small and task-specific.
Normal neural inference can return plausible but irrelevant neighbors. Scores
are not answer probabilities. Exact command syntax still comes from Clap help.

## Distribution

The model is pinned to an immutable upstream revision and hash-checked before
development-time quantization. Matrix weights are stored in symmetric 4-bit
groups of 32 with F16 scales; 1-D weights retain F32. They expand to F32 once at
load. Inference still runs all attention/feed-forward layers. No build script
fetches assets. `include_bytes!` embeds the weights and tokenizer in the final
binary. Separate crate packaging keeps each compressed publication within
crates.io's size limit. Prepared corpora avoid re-encoding documentation on
every short-lived CLI invocation.

## Primary sources

- [Clap suggestions implementation](https://github.com/clap-rs/clap/blob/v4.6.7/clap_builder/src/parser/features/suggestions.rs)
- [Clap command introspection](https://docs.rs/clap/4.6.7/clap/struct.Command.html)
- [Upstream model, architecture, pooling and Apache-2.0 license](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/tree/4ca70771034acceecb2e72475f72050fcdde4ddc)
- [Candle BERT implementation](https://github.com/huggingface/candle/blob/0.11.0/candle-transformers/src/models/bert.rs)
- [Model2Vec Rust](https://github.com/MinishLab/model2vec-rs)
- [FastEmbed Rust](https://github.com/Anush008/fastembed-rs)
- [RTen](https://github.com/robertknight/rten), [Tract](https://github.com/sonos/tract)
- [Ontology's lexical/embedding design](https://github.com/Gonzih/ontology)
- [Sentence Transformers semantic retrieval](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [Cargo publishing](https://doc.rust-lang.org/cargo/reference/publishing.html)
