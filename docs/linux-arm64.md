# Linux ARM64 portability

The patch line is pocket-encoder 0.1.2 and clap-how-to 0.1.5, based on the
published 0.1.1/0.1.4 source (`2ad2fd1`). Unrelated dynamic-source work on main
is outside this patch. Existing search APIs, PKTIDX01 storage and full ModelInfo
are retained.

The original dependency path was clap-how-to → pocket-encoder → Candle 0.11 →
GEMM 0.19. Its FP16 vector assembly wrappers omit `target_feature(fp16)`.
Generic aarch64 Linux does not enable that extension by default. A library's
local Cargo configuration cannot change its downstream consumer's build flags.

Linux ARM64 now executes the same three-layer MiniLM with F32-only GEMM,
GELU-erf, F32 layer normalization, attention masking and the existing tokenizer,
quantized-weight expansion and normalized pooling. Candle remains the backend
on other targets. No global CPU-feature flags, modified registry sources,
filesystem model loading or fallback to lexical search are used.

The Mac comparison test runs both backends on nine inputs (including empty
tokenization, Unicode and 128-token truncation), bounds normalized coordinate
error at 2e-5 and compares four top-five cosine rankings. Existing independent
PyTorch reference, index round-trip, fingerprint and help tests remain in place.
Weights/model identity are unchanged; cross-platform byte equality is not promised.

Local Linux verification uses the already installed Dory 0.4.5 engine on Apple
Silicon, official `rust:1.99-bookworm`, Rust 1.99.1 / LLVM 23.1.1,
`aarch64-unknown-linux-gnu`, no RUSTFLAGS. The unmodified Ontology consumer failed
with 11 `instruction requires: fullfp16` errors and exit 101. The patched upstream
workspace's all-feature/all-target debug tests pass. Release, packaged downstream
and CI evidence is recorded in the PR as it completes.

CI now includes the native `ubuntu-24.04-arm` runner and tests debug and release
with all features. The original-weight `reference` developer exporter still
requires Candle on a non-Linux-ARM64 host; bundled inference works on Linux ARM64.
