use clap_how_to::{Document, Index, Mode, Source};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input: serde_json::Value =
        serde_json::from_str(include_str!("../benchmarks/retrieval.json"))?;
    let documents = input["documents"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let name = row[0].as_str().unwrap();
            Document {
                title: name.into(),
                text: row[1].as_str().unwrap().into(),
                source: Source::Command {
                    command: name.into(),
                    help: format!("{name} --help"),
                },
            }
        })
        .collect();
    let start = Instant::now();
    let index = Index::new(documents)?;
    let prepare_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut modes = Vec::new();
    for (name, mode) in [
        ("lexical", Mode::Lexical),
        ("semantic", Mode::Semantic),
        ("hybrid", Mode::Hybrid),
    ] {
        let start = Instant::now();
        let mut rows = Vec::new();
        let (mut top1, mut top5, mut reciprocal) = (0, 0, 0.0);
        for row in input["queries"].as_array().unwrap() {
            let query = row[0].as_str().unwrap();
            let expected = row[1].as_str().unwrap();
            let hits = index.search_with_mode(query, 5, mode)?;
            let position = hits.iter().position(|hit| hit.document.title == expected);
            if position == Some(0) {
                top1 += 1;
            }
            if let Some(p) = position {
                top5 += 1;
                reciprocal += 1.0 / (p + 1) as f64;
            }
            rows.push(serde_json::json!({"query":query,"expected":expected,"rank":position.map(|p|p+1),"hits":hits}));
        }
        modes.push(serde_json::json!({"mode":name,"queries":rows.len(),"top1":top1,"top5":top5,"mrr_at_5":reciprocal/rows.len() as f64,"elapsed_ms":start.elapsed().as_secs_f64()*1000.0,"results":rows}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"model":clap_how_to::model_info(),"prepare_ms":prepare_ms,"modes":modes})
        )?
    );
    Ok(())
}
