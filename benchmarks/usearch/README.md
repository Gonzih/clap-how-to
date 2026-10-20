# USearch experiment

Measured 2026-10-20 on Apple M4, Rust 1.99.0, release builds, default target CPU.
USearch 2.26.5 uses NEON and F32 inner-product distance on normalized vectors.
The standalone Cargo workspace keeps USearch out of both production crates.
Inputs, per-query outcomes, p50/p95 timings and model identity are in
[`results.json`](results.json). Fresh queries are in [`probes.json`](probes.json).
The initial experiment is preserved below; the implemented optional index is
measured in the [pocket-encoder follow-up](#pocket-encoder-follow-up).

`pocket-encoder` works for suggestions. These results do not justify making
USearch its core dependency. An optional index API could be useful, but an exact
Rust scan remains adequate at the current 4,096-document limit. Binary vector
storage offers a much larger startup improvement than changing search engines.

## Embedding quality

All modes search the same 30 command descriptions. The existing queries were
used during development. The fresh probes were written before running this
experiment and were not tuned after seeing results. They are still handwritten
queries over the same small corpus, not an independent held-out benchmark.

| Queries | Retrieval | Correct first | Correct within five |
| --- | --- | ---: | ---: |
| Existing 30 | BM25 | 3/30 | 9/30 |
| Existing 30 | Bundled semantic | 19/30 | 27/30 |
| Existing 30 | Hybrid | 16/30 | 28/30 |
| Fresh 30 | BM25 | 12/30 | 19/30 |
| Fresh 30 | Bundled semantic | 20/30 | 29/30 |
| Fresh 30 | Original F32 semantic | 21/30 | 29/30 |
| Fresh 30 | Hybrid | 18/30 | 28/30 |

Across 90 texts, bundled versus original F32 embeddings have mean cosine
0.97078 and minimum 0.95563. The original weights, config and tokenizer match
the SHA-256 values in `pocket-encoder/assets/provenance.json`. This comparison
uses the existing native F32 reference implementation; the separate encoder
tests also pass against two independent PyTorch reference vectors.

The model can confuse opposite operations: “pause interruptions from the other
bots” ranks `unmute` before `mute`; restoring can rank below backing up.
“troubleshoot why the remote endpoint is unreachable” misses `doctor` in five.
USearch exact search preserves every measured first choice and top-k neighbor
set. It cannot fix these embedding errors. Tiny numerical differences reorder
some close neighbors at k=100 in the synthetic corpora.

Warm question encoding: 2.66 ms median. Across 20 fresh processes, model load
takes 21.69 ms median and first question encoding 3.18 ms; their combined median
is 24.84 ms. Timings exclude process launch. They are local measurements, not
latency guarantees.

## Vector search

Median microseconds per lookup, fetching 100 candidates as hybrid retrieval
does (all 30 when the corpus is smaller). Five measured passes over 60 queries,
one warm-up pass, 300 samples per case; vector search only, excluding encoding,
BM25, fusion, document grouping and the production final ordering pass.

Only 30 rows are real embeddings. Larger corpora contain seeded normalized
perturbations clustered around those rows. They measure scaling and stress
approximate search; they do not establish accuracy on a real large corpus.

| Vectors | Rust full scan + sort | USearch exact | HNSW defaults | HNSW neighbor recall |
| ---: | ---: | ---: | ---: | ---: |
| 30 | 3.67 µs | 3.42 µs | 4.46 µs | 100% |
| 512 | 64.83 µs | 51.71 µs | 41.58 µs | 98.75% |
| 4,096 | 537.88 µs | 349.62 µs | 49.46 µs | 36.88% |
| 32,768 | 4,393.00 µs | 2,409.17 µs | 47.67 µs | 10.43% |

Exact search has 100% neighbor-set recall in every case. At k=5 and 4,096 rows,
the Rust scan takes 538.83 µs versus 245.38 µs for USearch exact. Default HNSW
uses connectivity=16, expansion_add=128 and expansion_search=64. Raising search
expansion to 1,024 gives 70.95% recall at k=100 and 4,096 rows in 351.33 µs;
at 32,768 rows it gives 31.55% in 443.25 µs. This clustered stress test did not
find an acceptable approximate-search setting; it is not a general HNSW verdict.

## Startup and placement

At 4,096 vectors, building the USearch index costs 372 ms. Loading its prebuilt
6.90 MB artifact from memory costs 0.57 ms. USearch reports about 16 MiB for
its allocator usage at this size, including reserved memory; this is not an
RSS measurement. Keeping the existing prepared vectors would retain their
additional 6 MiB payload.

Decoding vectors alone from JSON takes **25.44 ms** (19.23 MB). Decoding the
same coordinates from packed little-endian F32 takes **0.139 ms** (6.29 MB),
without USearch. Both exclude document decoding, normalization checks, BM25
construction and disk I/O. Packed F32 is only an experiment here; no production
serialization format changed. Replacing JSON with a USearch artifact would
also avoid JSON decoding, so that benefit must not be credited to ANN search.

If a persistent application needs many searches over larger real corpora,
USearch becomes worth another benchmark. For the current CLI, keep the encoder
focused on embeddings. Any index API should be optional; the results favor
starting with exact scanning and compact storage. No production API or
dependency changed in this experiment.

## Reproduce

From the repository root:

```sh
cargo test --workspace --release --locked
cargo run --release --locked --manifest-path benchmarks/usearch/Cargo.toml \
  --target-dir target -- target/usearch-benchmark
```

For the optional F32 comparison, use the existing original weights under
`.kb/runs/research/neural`; no weights are downloaded by this benchmark. Verify
the three upstream hashes against `pocket-encoder/assets/provenance.json` first.

```sh
cargo run --release --locked -p pocket-encoder --example reference -- \
  .kb/runs/research/neural target/usearch-benchmark/reference-texts.json \
  > target/usearch-benchmark/upstream.json
cargo build --release --locked -p pocket-encoder --example embed
python3 benchmarks/usearch/compare_reference.py target/usearch-benchmark \
  target/release/examples/embed
```

Verification: workspace release tests passed (seven integration tests and one
doctest); benchmark, F32 reference, formatting and strict Clippy checks exited 0.
All 90 compared vectors
were finite and 384-dimensional; bundled vectors were unit length. All measured
exact-search neighbor sets and first choices matched the Rust scan. Production
source, manifests and lockfile were left untouched. Real large-corpus retrieval,
other hardware, binary-size changes and full application startup were not measured.

## Pocket-encoder follow-up

Implemented the optional `search` feature without adding dependencies:

- `pocket-encoder/src/index.rs:45` — validate model identity and vectors, then
  pack coordinates into one contiguous F32 allocation.
- `pocket-encoder/src/index.rs:105` — exact scan, partial selection and sorting
  of only the requested best rows, with stable row-ID ties.
- `pocket-encoder/src/index.rs:138` — versioned model metadata and little-endian
  F32 serialization; `:154` validates bounds, exact length, identity and norms
  on load without initializing the encoder.

The updated benchmark measures the public API, including query-vector validation
and complete binary-load validation. Raw results are in
[`pocket-index-results.json`](pocket-index-results.json). At 4,096 synthetic
vectors, Apple M4 release medians were:

| Operation | Median |
| --- | ---: |
| JSON vector decoding alone | 26.131 ms |
| `Index::from_bytes`, including validation | 0.617 ms |
| Original scan and full sort, 100 candidates | 0.543 ms |
| `Index::search_embedding`, 100 candidates | 0.488 ms |

The binary artifact is 6,291,932 bytes versus 19,231,499 bytes of vector JSON.
Across 30, 512 and 4,096 rows, all 60 queries at k=5 and k=100 produced identical
ordered results to the original full-sort baseline. The encoder's fresh-probe
results remain 20/30 first and 29/30 within five. The feature keeps the 4,096-row
bound; the 32,768-row experiment measures the older backends only.

Verification commands exited 0: default workspace release tests; workspace
release tests with all features (12 integration tests and one doctest); strict
Clippy for all workspace targets/features and for the benchmark; both formatting
checks; and the runnable `search` example. Tests cover exact selection and ties,
empty corpora, bit-preserving binary round trips, wrong model identities,
invalid vectors, malformed headers, truncation, oversized counts and trailing
bytes. Changes are local and uncommitted; no release was performed.
