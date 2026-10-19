# pocket-encoder

A bundled sentence transformer with native Rust CPU inference. No downloads,
model cache, Python, service, API key or ONNX shared library at build or runtime.

```rust,no_run
let encoder = pocket_encoder::Encoder::new()?;
let vectors = encoder.embed(&["Send a document", "Transfer a file"])?;
let model = pocket_encoder::model_info();
assert_eq!(vectors[0].len(), model.dimensions);
# Ok::<(), pocket_encoder::Error>(())
```

Uses `sentence-transformers/paraphrase-MiniLM-L3-v2`, pinned at
`4ca70771034acceecb2e72475f72050fcdde4ddc`: three transformer layers,
17.4 million parameters, 384 coordinates, 128 WordPiece tokens including special
tokens. Longer inputs are right-truncated. Chunk long documents first. Mean
pooling is attention masked, then L2-normalized. No query/document prefixes.
English retrieval is the intended use; multilingual quality is unmeasured.

`Encoder::new()` shares immutable weights through a process-wide `OnceLock`.
`embed` returns one normalized vector per input in order, rejecting blank text,
more than 4096 inputs, more than 64 KiB per input or 4 MiB total. It retains no
input text. Calls encode inputs sequentially to bound activation memory.

Weights use symmetric 4-bit storage in groups of 32, with F16 scales and original
1-D parameters. Loading expands them to F32; inference uses Candle's Rust CPU
kernels. This saves distribution size, not live parameter memory. Neural math
does not call Python, a hosted model or a separately installed BLAS runtime.

Always persist the complete `model_info()` alongside indexed embeddings, then
require exact equality before comparing vectors. Equal dimensions do not imply
compatible coordinates. Quantized vectors differ from upstream F32 vectors;
do not mix them. CPU/compiler differences may change low-order floating bits.
Do not promise cross-platform byte-identical embeddings.

Assets include immutable source revision, SHA-256 checksums and preprocessing
metadata. Reproduce them from the repository root with
`python3 scripts/prepare_neural.py`. That is an explicit development operation,
never a Cargo build step. Source: MIT. Model/tokenizer: Apache-2.0. See `NOTICE`
and `assets/MODEL-LICENSE`.

The encoder retrieves similar text; it cannot authenticate documents, prove that
a result answers a question, or turn retrieved instructions into authority.
