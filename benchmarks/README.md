# Measurements

See the [USearch experiment](usearch/README.md) for vector-search, loading-format
and encoder-quality comparisons measured on 2026-10-20.

Apple M4, Rust 1.99.0, release build, 2026-10-19 UTC. The complete inputs and
per-query ranks are in `retrieval.json` and `results.json`. This is a handwritten
30-query English task smoke test, not a held-out evaluation or general model
leaderboard. We used it while developing the retrieval rules.

| Mode | Correct first | Correct within five | MRR@5 |
| --- | ---: | ---: | ---: |
| BM25 | 3/30 | 9/30 | 0.184 |
| Neural cosine | 19/30 | 27/30 | 0.744 |
| Hybrid | 16/30 | 28/30 | 0.686 |

Hybrid favors coverage among five suggested sources. It does not win every
metric. It fuses at most 100 entries per ranking with weights neural=2, BM25=1
and RRF constant 10. BM25 uses a small English function-word stop list, not
product-specific synonyms. `Mode::Semantic` and `Mode::Lexical` expose ablations.

Across 20 fresh processes encoding one short question, model load median was
37.5 ms (p95 49.4 ms); inference median 6.8 ms (p95 12.0 ms). The 30-document
corpus took 277 ms to encode including initial model load. Ship a prepared index
to avoid corpus inference on each invocation. These are local measurements,
not latency bounds on other hardware. Debug builds are much slower.

Quantization changes vectors. Across 64 example inputs, bundled quantized versus
upstream F32 PyTorch embeddings had mean cosine 0.9707 and minimum 0.9546.
An independent native F32 Candle run matched the PyTorch reference at mean
0.99999999, minimum 0.99999989. Thus the observed drift comes from quantization,
not an unnoticed pooling/architecture mismatch. Do not mix upstream vectors
with this model's vectors. Two independent PyTorch reference vectors are retained
in the encoder tests. This fidelity check does not establish retrieval quality
for arbitrary documents or languages.

```sh
cargo run --release --example evaluate > evaluation.json
cargo run --release -p pocket-encoder --example embed -- "your question"
# Explicit optional development download, never a build step:
python3 scripts/prepare_neural.py
cargo run --release -p pocket-encoder --example reference -- \
  .kb/runs/research/neural texts.json
```

`reference` accepts a JSON string array and emits normalized upstream-F32 vectors.
The original comparison used Transformers 4.57.6 and PyTorch in an isolated
development environment, with pinned local weights, max_length 128, masked mean
pooling, L2 normalization and eval mode. Runtime users need none of that Python
environment. `pocket-encoder` packages at about 8.7 MiB compressed; exact package
size depends on included source and Cargo metadata.

During integration we also compared the same-size retrieval-trained
`TaylorAI/bge-micro-v2` (revision `3edf6d7de0faa426b09780416fe61009f26ae589`)
in upstream and group-32 Q4 forms. It improved inbox retrieval but worsened file
and focus questions on the same seven-question ICQ development snapshot; it did
not establish an overall improvement. We retained the published MiniLM encoder,
clarified actual command intent descriptions, and balanced document chunks to
avoid orphaned fragments. Integration tests require the intended command among
five suggestions; nearest-neighbor search still does not guarantee an answer.
