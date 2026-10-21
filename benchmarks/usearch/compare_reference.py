"""Compare bundled embeddings/retrieval with the local upstream F32 model.

Run after the benchmark and pocket-encoder's reference example; see README.md.
Uses only Python's standard library. No downloads or model changes.
"""

import json
import math
import pathlib
import subprocess
import sys

run = pathlib.Path(sys.argv[1])
vectors = json.loads((run / "vectors.json").read_text())
upstream = json.loads((run / "upstream.json").read_text())["embeddings"]
bundled = vectors["documents"] + vectors["queries"]
assert len(bundled) == len(upstream)
assert all(len(row) == 384 for row in upstream + bundled)


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(
        sum(x * x for x in a) * sum(y * y for y in b)
    )


def metrics(documents, queries, labels):
    top1 = top5 = 0
    reciprocal = 0.0
    results = []
    for vector, (query, expected) in zip(queries, labels):
        # Stable row ties, equivalent cosine on normalized reference vectors.
        scores = [cosine(vector, doc) for doc in documents]
        hits = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:5]
        titles = [vectors["titles"][i] for i in hits]
        rank = titles.index(expected) + 1 if expected in titles else None
        top1 += rank == 1
        top5 += rank is not None
        reciprocal += 1 / rank if rank else 0
        results.append(dict(query=query, expected=expected, rank=rank, hits=titles))
    return dict(
        queries=len(labels), top1=top1, top5=top5,
        mrr_at_5=reciprocal / len(labels), results=results,
    )


count = len(vectors["documents"])
similarities = [cosine(a, b) for a, b in zip(upstream, bundled)]
report = json.loads((run / "results.json").read_text())
report["upstream_f32"] = dict(
    reference="pocket-encoder/examples/reference.rs; local original safetensors; same tokenizer and preprocessing",
    embeddings=len(similarities), mean_cosine=sum(similarities) / len(similarities),
    min_cosine=min(similarities),
    quality=[
        dict(suite=name, **metrics(upstream[:count], upstream[count:][lo:hi], vectors["labels"][lo:hi]))
        for name, lo, hi in [("existing", 0, 30), ("fresh_probes", 30, 60)]
    ],
)
if len(sys.argv) > 2:
    samples = []
    for _ in range(20):
        result = json.loads(subprocess.check_output([
            sys.argv[2], "protect me from losing everything",
        ]))
        samples.append({key: result[key] for key in ["load_ms", "encode_ms"]})

    def milliseconds(values):
        values = sorted(values)
        return dict(p50_ms=values[(len(values) - 1) // 2],
                    p95_ms=values[math.ceil(len(values) * .95) - 1])

    report["fresh_processes"] = dict(
        processes=20, query="protect me from losing everything", samples=samples,
        load_ms=milliseconds([row["load_ms"] for row in samples]),
        encode_ms=milliseconds([row["encode_ms"] for row in samples]),
        load_plus_encode=milliseconds([row["load_ms"] + row["encode_ms"] for row in samples]),
    )
(run / "results.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: report[key] for key in ["query_encode", "upstream_f32"]}, indent=2))
