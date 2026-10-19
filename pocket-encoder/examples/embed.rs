//! JSON output and timings for reproducible CPU measurements.
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let texts = std::env::args().skip(1).collect::<Vec<_>>();
    let start = Instant::now();
    let encoder = pocket_encoder::Encoder::new()?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let vectors = encoder.embed(&texts)?;
    println!(
        "{}",
        serde_json::json!({"model":pocket_encoder::model_info(),"load_ms":load_ms,"encode_ms":start.elapsed().as_secs_f64()*1000.0,"embeddings":vectors})
    );
    Ok(())
}
