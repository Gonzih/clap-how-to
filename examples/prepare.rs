//! Read an explicit developer-supplied Document JSON array, emit a prepared index.
use std::io::Read;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("usage: prepare documents.json > index.json")?;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err("document JSON exceeds 16 MiB".into());
    }
    let documents = serde_json::from_slice(&bytes)?;
    let index = clap_how_to::Index::new(documents)?;
    println!("{}", serde_json::to_string(index.prepared())?);
    Ok(())
}
