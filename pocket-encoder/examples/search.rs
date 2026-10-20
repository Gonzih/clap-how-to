//! Encode once during development, then ship the binary index and your labels.
use pocket_encoder::Index;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let labels = ["backup", "delete", "login"];
    let index = Index::new(&[
        "backup: Save a recoverable snapshot of all project data.",
        "delete: Permanently remove a snapshot and its stored contents.",
        "login: Authenticate the current user using browser OAuth authorization.",
    ])?;
    let bytes = index.to_bytes();
    // Applications can save these bytes and restore include_bytes!(...) instead.
    let restored = Index::from_bytes(&bytes)?;
    let query = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "protect me from losing everything".into());
    for hit in restored.search(&query, 3)? {
        println!("{}\t{:.3}", labels[hit.index], hit.score);
    }
    Ok(())
}
