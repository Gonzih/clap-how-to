//! Reproducible experiment, deliberately separate from the production workspace.
use clap_how_to::{Document, Encoder, Index, Mode, Source};
use pocket_encoder::Index as PocketIndex;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{error::Error, hint::black_box, path::Path, process::Command, time::Instant};
use usearch::{Index as UIndex, IndexOptions, MetricKind, ScalarKind};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Suite {
    documents: Vec<[String; 2]>,
    queries: Vec<[String; 2]>,
}

fn command(program: &str, args: &[&str]) -> String {
    let output = Command::new(program).args(args).output().unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().into()
}

fn distribution(mut samples: Vec<f64>) -> Value {
    samples.sort_by(f64::total_cmp);
    json!({
        "samples": samples.len(),
        "p50_us": samples[(samples.len() - 1) / 2],
        "p95_us": samples[(samples.len() * 95).div_ceil(100) - 1],
    })
}

// Match src/index.rs: F32 dot products, clamping, full sort, stable row-ID ties.
fn scan(rows: &[Vec<f32>], query: &[f32], count: usize) -> Vec<usize> {
    let mut scores = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let score = row.iter().zip(query).map(|(a, b)| a * b).sum::<f32>();
            (i, score.clamp(-1.0, 1.0) as f64)
        })
        .collect::<Vec<_>>();
    scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scores.into_iter().take(count).map(|(i, _)| i).collect()
}

fn neighbors(index: &UIndex, query: &[f32], count: usize, exact: bool) -> Vec<usize> {
    let found = if exact {
        index.exact_search(query, count)
    } else {
        index.search(query, count)
    }
    .unwrap();
    found.keys.into_iter().map(|key| key as usize).collect()
}

fn timed_search(queries: &[Vec<f32>], mut search: impl FnMut(&[f32]) -> Vec<usize>) -> Value {
    for query in queries {
        black_box(search(black_box(query)));
    }
    let mut samples = Vec::new();
    for _ in 0..5 {
        for query in queries {
            let started = Instant::now();
            let found = search(black_box(query));
            samples.push(started.elapsed().as_secs_f64() * 1e6);
            black_box(found);
        }
    }
    distribution(samples)
}

fn recall(expected: &[Vec<usize>], actual: &[Vec<usize>]) -> f64 {
    let found = expected
        .iter()
        .zip(actual)
        .map(|(a, b)| b.iter().filter(|key| a.contains(key)).count())
        .sum::<usize>();
    found as f64 / expected.iter().map(Vec::len).sum::<usize>() as f64
}

fn vector_decoding(rows: &[Vec<f32>]) -> Value {
    let json = serde_json::to_vec(rows).unwrap();
    let binary = rows
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let mut timings = Vec::new();
    for packed in [false, true] {
        let mut samples = Vec::new();
        for round in 0..11 {
            let started = Instant::now();
            let decoded: Vec<Vec<f32>> = if packed {
                binary
                    .as_chunks::<{ 384 * 4 }>()
                    .0
                    .iter()
                    .map(|row| {
                        row.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|v| f32::from_le_bytes(*v))
                            .collect()
                    })
                    .collect()
            } else {
                serde_json::from_slice(&json).unwrap()
            };
            if round > 0 {
                samples.push(started.elapsed().as_secs_f64() * 1e6);
            }
            assert_eq!(decoded.len(), rows.len());
            black_box(decoded);
        }
        timings.push(distribution(samples));
    }
    json!({"json_bytes":json.len(),"packed_f32_bytes":binary.len(),
        "json":timings[0],"packed_f32":timings[1],
        "method":"In-memory decoding of vectors only into Vec<Vec<f32>>; no document loading, norm validation, lexical indexing or disk I/O. Packed F32 is an experimental comparison, not an implemented production format."})
}

// Synthetic normalized clusters around the 30 real vectors, for scaling only.
// Seed and row order are fixed. These rows make no semantic-quality claim.
fn padded(base: &[Vec<f32>], count: usize) -> Vec<Vec<f32>> {
    let mut rows = base.to_vec();
    let mut state = 0x6a09e667f3bcc909_u64;
    while rows.len() < count {
        let mut row = base[rows.len() % base.len()]
            .iter()
            .map(|v| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                v + ((state >> 40) as f32 / (1_u32 << 24) as f32 - 0.5) * 0.1
            })
            .collect::<Vec<_>>();
        let norm = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        for v in &mut row {
            *v = (*v as f64 / norm) as f32;
        }
        rows.push(row);
    }
    rows
}

fn quality(index: &Index, queries: &[[String; 2]], mode: Mode) -> Result<Value> {
    let mut results = Vec::new();
    let mut times = Vec::new();
    let (mut top1, mut top5, mut reciprocal) = (0, 0, 0.0);
    for [query, expected] in queries {
        let started = Instant::now();
        let hits = index.search_with_mode(query, 5, mode)?;
        times.push(started.elapsed().as_secs_f64() * 1e6);
        let rank = hits
            .iter()
            .position(|hit| &hit.document.title == expected)
            .map(|i| i + 1);
        if rank == Some(1) {
            top1 += 1;
        }
        if let Some(rank) = rank {
            top5 += 1;
            reciprocal += 1.0 / rank as f64;
        }
        results.push(json!({"query":query,"expected":expected,"rank":rank,
            "hits":hits.iter().map(|h| &h.document.title).collect::<Vec<_>>()}));
    }
    Ok(json!({"queries":queries.len(),"top1":top1,"top5":top5,
        "mrr_at_5":reciprocal / queries.len() as f64,"latency":distribution(times),"results":results}))
}

fn main() -> Result<()> {
    let output_arg = std::env::args()
        .nth(1)
        .expect("usage: retrieval-benchmark OUTPUT_DIRECTORY");
    let output = Path::new(&output_arg);
    std::fs::create_dir_all(output)?;
    let suite: Suite = serde_json::from_str(include_str!("../../retrieval.json"))?;
    let probes: Vec<[String; 2]> = serde_json::from_str(include_str!("../probes.json"))?;
    let queries = suite
        .queries
        .iter()
        .chain(&probes)
        .cloned()
        .collect::<Vec<_>>();
    let documents = suite
        .documents
        .iter()
        .map(|[title, text]| Document {
            title: title.clone(),
            text: text.clone(),
            source: Source::Command {
                command: title.clone(),
                help: format!("{title} --help"),
            },
        })
        .collect::<Vec<_>>();

    let started = Instant::now();
    let encoder = Encoder::new()?;
    let load_ms = started.elapsed().as_secs_f64() * 1e3;
    let started = Instant::now();
    let index = Index::new(documents)?;
    let corpus_encode_ms = started.elapsed().as_secs_f64() * 1e3;
    let mut query_vectors = Vec::new();
    let mut encode_times = Vec::new();
    for [query, _] in &queries {
        let started = Instant::now();
        query_vectors.push(encoder.embed(&[query])?.remove(0));
        encode_times.push(started.elapsed().as_secs_f64() * 1e6);
    }
    let base = &index.prepared().embeddings;
    let texts = suite
        .documents
        .iter()
        .map(|[title, text]| format!("{title}: {text}"))
        .chain(queries.iter().map(|[query, _]| query.clone()))
        .collect::<Vec<_>>();
    std::fs::write(
        output.join("reference-texts.json"),
        serde_json::to_vec(&texts)?,
    )?;
    std::fs::write(
        output.join("vectors.json"),
        serde_json::to_vec(&json!({
            "documents":base,"queries":query_vectors,"labels":queries,
            "titles":suite.documents.iter().map(|row| &row[0]).collect::<Vec<_>>()
        }))?,
    )?;

    let mut suites = Vec::new();
    for (name, rows) in [("existing", &suite.queries), ("fresh_probes", &probes)] {
        let mut modes = serde_json::Map::new();
        for (name, mode) in [
            ("lexical", Mode::Lexical),
            ("semantic", Mode::Semantic),
            ("hybrid", Mode::Hybrid),
        ] {
            modes.insert(name.into(), quality(&index, rows, mode)?);
        }
        suites.push(json!({"suite":name,"modes":modes}));
    }

    let mut scales = Vec::new();
    for count in [30, 512, 4096, 32768] {
        eprintln!("benchmarking {count} vectors");
        let rows = padded(base, count);
        let pocket = if count <= 4096 {
            Some(PocketIndex::from_embeddings(
                clap_how_to::model_info(),
                rows.clone(),
            )?)
        } else {
            None
        };
        let pocket_storage = if let Some(index) = &pocket {
            let bytes = index.to_bytes();
            let mut samples = Vec::new();
            for round in 0..11 {
                let started = Instant::now();
                let restored = PocketIndex::from_bytes(&bytes)?;
                if round > 0 {
                    samples.push(started.elapsed().as_secs_f64() * 1e6);
                }
                assert_eq!(restored.len(), count);
                black_box(restored);
            }
            json!({"serialized_bytes":bytes.len(),"validated_load":distribution(samples)})
        } else {
            Value::Null
        };
        let options = IndexOptions {
            dimensions: 384,
            metric: MetricKind::IP,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let started = Instant::now();
        let native = UIndex::new(&options)?;
        native.reserve(count)?;
        for (i, row) in rows.iter().enumerate() {
            native.add(i as u64, row)?;
        }
        let build_ms = started.elapsed().as_secs_f64() * 1e3;
        let saved = output.join(format!("{count}.usearch"));
        native.save(saved.to_str().unwrap())?;
        let bytes = std::fs::read(&saved)?;
        let mut load_times = Vec::new();
        for _ in 0..10 {
            let started = Instant::now();
            let restored = UIndex::new(&options)?;
            restored.load_from_buffer(&bytes)?;
            load_times.push(started.elapsed().as_secs_f64() * 1e6);
            assert_eq!(restored.size(), count);
            black_box(restored);
        }
        let mut searches = Vec::new();
        for k in [5, 100] {
            let k_actual = k.min(count);
            let exact = query_vectors
                .iter()
                .map(|q| scan(&rows, q, k_actual))
                .collect::<Vec<_>>();
            for (name, native_exact) in [
                ("rust_scan", None),
                ("usearch_exact", Some(true)),
                ("usearch_hnsw", Some(false)),
            ] {
                let search = |q: &[f32]| match native_exact {
                    None => scan(&rows, q, k_actual),
                    Some(exact) => neighbors(&native, q, k_actual, exact),
                };
                let actual = query_vectors.iter().map(|q| search(q)).collect::<Vec<_>>();
                searches.push(json!({"backend":name,"requested_k":k,"returned_k":k_actual,
                    "recall_at_k":recall(&exact, &actual),
                    "top1_agreement":exact.iter().zip(&actual).filter(|(a,b)|a.first()==b.first()).count() as f64 / exact.len() as f64,
                    "ordered_match_rate":exact.iter().zip(&actual).filter(|(a,b)|a==b).count() as f64 / exact.len() as f64,
                    "latency":timed_search(&query_vectors, search)}));
            }
            if let Some(index) = &pocket {
                let search = |query: &[f32]| {
                    index
                        .search_embedding(index.model(), query, k_actual)
                        .unwrap()
                        .into_iter()
                        .map(|hit| hit.index)
                        .collect::<Vec<_>>()
                };
                let actual = query_vectors
                    .iter()
                    .map(|query| search(query))
                    .collect::<Vec<_>>();
                assert_eq!(actual, exact);
                searches.push(json!({"backend":"pocket_encoder","requested_k":k,"returned_k":k_actual,
                    "recall_at_k":recall(&exact, &actual),"top1_agreement":1.0,"ordered_match_rate":1.0,
                    "latency":timed_search(&query_vectors, search)}));
            }
        }
        let default_expansion = native.expansion_search();
        if count >= 4096 {
            native.change_expansion_search(1024);
            for k in [5, 100] {
                let exact = query_vectors
                    .iter()
                    .map(|q| scan(&rows, q, k))
                    .collect::<Vec<_>>();
                let search = |q: &[f32]| neighbors(&native, q, k, false);
                let actual = query_vectors.iter().map(|q| search(q)).collect::<Vec<_>>();
                searches.push(json!({"backend":"usearch_hnsw_expansion_1024","requested_k":k,"returned_k":k,
                    "recall_at_k":recall(&exact, &actual),
                    "top1_agreement":exact.iter().zip(&actual).filter(|(a,b)|a.first()==b.first()).count() as f64 / exact.len() as f64,
                    "ordered_match_rate":exact.iter().zip(&actual).filter(|(a,b)|a==b).count() as f64 / exact.len() as f64,
                    "latency":timed_search(&query_vectors, search)}));
            }
        }
        scales.push(json!({"vectors":count,"synthetic":count>base.len(),"build_ms":build_ms,
            "load_from_memory":distribution(load_times),"serialized_bytes":bytes.len(),
            "usearch_memory_lower_bound_bytes":native.memory_usage(),"rust_vector_payload_bytes":count*384*4,
            "hardware_acceleration":native.hardware_acceleration(),"connectivity":native.connectivity(),
            "pocket_encoder_storage":pocket_storage,
            "vector_decoding":if count<=4096 { vector_decoding(&rows) } else { Value::Null },
            "expansion_add":native.expansion_add(),"default_expansion_search":default_expansion,"searches":searches}));
    }
    let report = json!({
        "measured_at":command("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "machine":command("sysctl", &["-n", "machdep.cpu.brand_string"]),
        "rust":command("rustc", &["--version"]),"git_head":command("git", &["rev-parse","HEAD"]),
        "model":clap_how_to::model_info(),"usearch_version":"2.26.5",
        "method":"Release, default target CPU; single caller; F32 normalized inner product; full-sort production scan; one warm-up pass, five measured passes over 60 queries; ANN defaults; 30 real embeddings, larger corpora seeded synthetic clusters; no semantic inference from synthetic rows. Fresh probes authored before first run, not a held-out dataset.",
        "model_load_ms":load_ms,"corpus_encode_ms":corpus_encode_ms,"query_encode":distribution(encode_times),
        "quality":suites,"scales":scales
    });
    std::fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", output.join("results.json").display());
    Ok(())
}
