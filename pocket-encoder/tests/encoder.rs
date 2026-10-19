use pocket_encoder::{Encoder, model_info};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[test]
fn finite_unit_contextual_deterministic_and_bounds() -> Result<(), Box<dyn std::error::Error>> {
    let encoder = Encoder::new()?;
    let vectors = encoder.embed(&[
        "Send a document",
        "Transfer a file",
        "A volcano erupts",
        "The dog chased the cat",
        "The cat chased the dog",
    ])?;
    for row in &vectors {
        assert_eq!(row.len(), 384);
        assert!(row.iter().all(|v| v.is_finite()));
        assert!((cosine(row, row) - 1.0).abs() < 0.0001);
    }
    assert!(cosine(&vectors[0], &vectors[1]) > cosine(&vectors[0], &vectors[2]) + 0.2);
    assert!(cosine(&vectors[3], &vectors[4]) < 0.999); // word order reaches attention layers
    assert_eq!(encoder.clone().embed(&["Send a document"])?[0], vectors[0]);
    assert!(encoder.embed(&[""]).is_err());
    assert!(encoder.embed(&["x".repeat(65537)]).is_err());
    assert!(encoder.embed(&vec!["text"; 4097]).is_err());
    assert!(encoder.embed(&vec!["x".repeat(65536); 65]).is_err());
    assert_eq!(
        model_info().upstream_revision,
        "4ca70771034acceecb2e72475f72050fcdde4ddc"
    );
    let prefix = "word ".repeat(200);
    assert_eq!(
        encoder.embed(&[format!("{prefix}cat"), format!("{prefix}dog")])?[0],
        encoder.embed(&[prefix])?[0]
    );
    Ok(())
}

#[test]
fn bundled_bytes_match_fingerprint() {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(include_bytes!("../assets/model.safetensors"));
    let encoded = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();
    assert_eq!(model_info().revision, format!("sha256:{encoded}"));
    let hash = Sha256::digest(include_bytes!("../assets/tokenizer.json"));
    assert_eq!(
        model_info().tokenizer_sha256,
        hash.iter().map(|b| format!("{b:02x}")).collect::<String>()
    );
}

#[test]
fn independent_pytorch_reference_bounds_quantization_drift()
-> Result<(), Box<dyn std::error::Error>> {
    let reference: serde_json::Value = serde_json::from_str(include_str!("reference.json"))?;
    let texts = reference["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect::<Vec<_>>();
    let vectors = Encoder::new()?.embed(&texts)?;
    for (actual, expected) in vectors
        .iter()
        .zip(reference["embeddings"].as_array().unwrap())
    {
        let expected = expected
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        assert!(
            cosine(actual, &expected) > 0.95,
            "quantization drift exceeded the measured reference bound"
        );
    }
    Ok(())
}
