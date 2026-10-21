#![cfg(feature = "search")]

use pocket_encoder::{Encoder, Index, ModelInfo, Neighbor, model_info};

fn unit(axis: usize) -> Vec<f32> {
    let mut row = vec![0.0; 384];
    row[axis] = 1.0;
    row
}

#[test]
fn selection_matches_full_sort_including_ties_and_limits() -> Result<(), Box<dyn std::error::Error>>
{
    let rows = (0..512)
        .map(|i| {
            let mut row = unit(i % 384);
            row[(i + 1) % 384] = 0.5;
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            row.iter_mut().for_each(|v| *v /= norm);
            row
        })
        .collect::<Vec<_>>();
    let mut query = unit(0);
    query[1] = 0.5;
    let norm = query.iter().map(|v| v * v).sum::<f32>().sqrt();
    query.iter_mut().for_each(|v| *v /= norm);
    let mut expected = rows
        .iter()
        .enumerate()
        .map(|(index, row)| Neighbor {
            index,
            score: row
                .iter()
                .zip(&query)
                .map(|(a, b)| a * b)
                .sum::<f32>()
                .clamp(-1.0, 1.0),
        })
        .collect::<Vec<_>>();
    expected.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.index.cmp(&b.index)));
    let index = Index::from_embeddings(model_info(), rows)?;
    assert_eq!(index.len(), 512);
    assert!(!index.is_empty());
    for limit in [0, 1, 5, 100, 511, 512, usize::MAX] {
        let hits = index.search_embedding(index.model(), &query, limit)?;
        assert_eq!(hits, expected[..limit.min(512)]);
    }
    assert_eq!(expected[0].index, 0);
    assert_eq!(expected[1].index, 384);
    Ok(())
}

#[test]
fn binary_roundtrip_preserves_coordinates_identity_and_ranking()
-> Result<(), Box<dyn std::error::Error>> {
    let mut row = unit(0);
    row[10] = -0.0;
    let index = Index::from_embeddings(model_info(), vec![row.clone(), unit(1)])?;
    let bytes = index.to_bytes();
    assert_eq!(&bytes[..8], b"PKTIDX01");
    assert_eq!(u32::from_le_bytes(bytes[12..16].try_into()?), 2);
    let model_len = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
    let stored: ModelInfo = serde_json::from_slice(&bytes[16..16 + model_len])?;
    assert_eq!(stored, model_info());
    assert_eq!(
        &bytes[16 + model_len..20 + model_len],
        &1.0_f32.to_le_bytes()
    );
    let restored = Index::from_bytes(&bytes)?;
    assert_eq!(restored.to_bytes(), bytes); // Includes the sign bit of -0.0.
    assert_eq!(restored.model(), index.model());
    assert_eq!(
        restored.search_embedding(restored.model(), &row, 2)?,
        index.search_embedding(index.model(), &row, 2)?
    );
    Ok(())
}

#[test]
fn rejects_invalid_vectors_and_coordinate_spaces() -> Result<(), Box<dyn std::error::Error>> {
    let index = Index::from_embeddings(model_info(), vec![unit(0)])?;
    for bad in [
        vec![],
        vec![0.0; 384],
        vec![1.0; 384],
        vec![f32::NAN; 384],
        vec![f32::INFINITY; 384],
        vec![1.0; 383],
    ] {
        assert!(Index::from_embeddings(model_info(), vec![bad.clone()]).is_err());
        assert!(index.search_embedding(index.model(), &bad, 1).is_err());
    }
    let mut wrong = model_info();
    wrong.tokenizer_sha256.push('x');
    assert!(Index::from_embeddings(wrong.clone(), vec![unit(0)]).is_err());
    assert!(index.search_embedding(&wrong, &unit(0), 1).is_err());
    assert!(Index::from_embeddings(model_info(), vec![unit(0); 4097]).is_err());
    Ok(())
}

#[test]
fn binary_loader_rejects_malformed_headers_lengths_and_values()
-> Result<(), Box<dyn std::error::Error>> {
    let bytes = Index::from_embeddings(model_info(), vec![unit(0)])?.to_bytes();
    for length in [0, 7, 15, 16, bytes.len() - 1] {
        assert!(Index::from_bytes(&bytes[..length]).is_err());
    }
    for (offset, replacement) in [(0, b'X'), (7, b'2'), (16, b'!')] {
        let mut bad = bytes.clone();
        bad[offset] = replacement;
        assert!(Index::from_bytes(&bad).is_err());
    }
    for offset in [8, 12] {
        let mut bad = bytes.clone();
        bad[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Index::from_bytes(&bad).is_err());
    }
    let mut bad = bytes.clone();
    bad.push(0);
    assert!(Index::from_bytes(&bad).is_err());
    let model_len = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
    for value in [f32::NAN, f32::INFINITY, 0.0, 2.0] {
        let mut bad = bytes.clone();
        bad[16 + model_len..20 + model_len].copy_from_slice(&value.to_le_bytes());
        assert!(Index::from_bytes(&bad).is_err());
    }
    let mut bad = bytes.clone();
    let mut model: ModelInfo = serde_json::from_slice(&bad[16..16 + model_len])?;
    model.revision.replace_range(7..8, "x");
    let model = serde_json::to_vec(&model)?;
    assert_eq!(model.len(), model_len);
    bad[16..16 + model_len].copy_from_slice(&model);
    assert!(Index::from_bytes(&bad).is_err());
    Ok(())
}

#[test]
fn text_search_and_empty_corpora() -> Result<(), Box<dyn std::error::Error>> {
    let empty = Index::new(&[] as &[&str])?;
    assert!(empty.is_empty());
    assert!(Index::from_bytes(&empty.to_bytes())?.is_empty());
    assert!(empty.search("a question", 5)?.is_empty());
    assert!(empty.search(" ", 5).is_err());
    assert!(empty.search(&"x".repeat(65537), 5).is_err());
    let labels = [
        "Save a recoverable copy of your data",
        "Permanently erase the stored data",
        "Authenticate a user",
    ];
    let index = Index::new(&labels)?;
    let bytes = index.to_bytes();
    let restored = Index::from_bytes(&bytes)?;
    let hits = restored.search("protect me from losing everything", 3)?;
    let query = Encoder::new()?
        .embed(&["protect me from losing everything"])?
        .remove(0);
    assert_eq!(
        hits,
        restored.search_embedding(restored.model(), &query, 3)?
    );
    assert_eq!(restored.search(labels[0], 1)?[0].index, 0);
    assert_eq!(hits, index.search("protect me from losing everything", 3)?);
    assert!(restored.search("a question", 0)?.is_empty());
    assert!(restored.search("", 0).is_err());
    Ok(())
}
