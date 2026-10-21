//! F32-only execution of the bundled three-layer BERT on generic Linux ARM64.
//! Uses the same GEMM F32 kernels as Candle, without compiling its FP16 backend.
use crate::{
    Error, inference,
    weights::{self, Weight},
};
use std::collections::HashMap;

const HIDDEN: usize = 384;
const HEADS: usize = 12;
const HEAD: usize = HIDDEN / HEADS;
const INTERMEDIATE: usize = 1536;

pub(crate) struct Model {
    weights: HashMap<String, Weight>,
}

impl Model {
    pub fn load() -> Result<Self, Error> {
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../assets/config.json")).map_err(inference)?;
        if config["hidden_size"] != HIDDEN
            || config["num_attention_heads"] != HEADS
            || config["num_hidden_layers"] != 3
            || config["intermediate_size"] != INTERMEDIATE
            || config["hidden_act"] != "gelu"
            || config["layer_norm_eps"] != 1e-12
        {
            return Err(inference("unsupported bundled MiniLM configuration"));
        }
        Ok(Self {
            weights: weights::load()?,
        })
    }

    fn weight(&self, name: &str, shape: &[usize]) -> Result<&[f32], Error> {
        let weight = self
            .weights
            .get(name)
            .ok_or_else(|| inference("missing bundled weight"))?;
        if weight.shape != shape {
            return Err(inference("invalid bundled weight shape"));
        }
        Ok(&weight.data)
    }

    fn linear(
        &self,
        input: &[f32],
        rows: usize,
        width: usize,
        output: usize,
        name: &str,
    ) -> Result<Vec<f32>, Error> {
        let weights = self.weight(&format!("{name}.weight"), &[output, width])?;
        let bias = self.weight(&format!("{name}.bias"), &[output])?;
        let mut result = matmul(input, rows, width, weights, output, true);
        for row in result.chunks_exact_mut(output) {
            for (value, bias) in row.iter_mut().zip(bias) {
                *value += bias;
            }
        }
        Ok(result)
    }

    fn norm(&self, values: &mut [f32], name: &str) -> Result<(), Error> {
        let scale = self.weight(&format!("{name}.weight"), &[HIDDEN])?;
        let bias = self.weight(&format!("{name}.bias"), &[HIDDEN])?;
        for row in values.chunks_exact_mut(HIDDEN) {
            // Match Candle's F32 layer-norm reduction and epsilon.
            let mut sum = 0.0f32;
            let mut sum2 = 0.0f32;
            for &v in row.iter() {
                sum += v;
                sum2 += v * v;
            }
            let mean = sum / HIDDEN as f32;
            let inverse = (sum2 / HIDDEN as f32 - mean * mean + 1e-12).sqrt().recip();
            for ((v, scale), bias) in row.iter_mut().zip(scale).zip(bias) {
                *v = (*v - mean) * inverse * scale + bias;
            }
        }
        Ok(())
    }

    pub fn pooled(&self, ids: &[u32], types: &[u32], mask: &[u32]) -> Result<Vec<f32>, Error> {
        let n = ids.len();
        if n == 0 || n > 128 || types.len() != n || mask.len() != n {
            return Err(inference("invalid token dimensions"));
        }
        let words = self.weight("embeddings.word_embeddings.weight", &[30522, HIDDEN])?;
        let positions = self.weight("embeddings.position_embeddings.weight", &[512, HIDDEN])?;
        let token_types = self.weight("embeddings.token_type_embeddings.weight", &[2, HIDDEN])?;
        let mut hidden = vec![0.0; n * HIDDEN];
        for (i, (&id, &kind)) in ids.iter().zip(types).enumerate() {
            if id >= 30522 || kind >= 2 {
                return Err(inference("invalid token id"));
            }
            for j in 0..HIDDEN {
                hidden[i * HIDDEN + j] = (words[id as usize * HIDDEN + j]
                    + token_types[kind as usize * HIDDEN + j])
                    + positions[i * HIDDEN + j];
            }
        }
        self.norm(&mut hidden, "embeddings.LayerNorm")?;
        for layer in 0..3 {
            let prefix = format!("encoder.layer.{layer}");
            let query = self.linear(
                &hidden,
                n,
                HIDDEN,
                HIDDEN,
                &format!("{prefix}.attention.self.query"),
            )?;
            let key = self.linear(
                &hidden,
                n,
                HIDDEN,
                HIDDEN,
                &format!("{prefix}.attention.self.key"),
            )?;
            let value = self.linear(
                &hidden,
                n,
                HIDDEN,
                HIDDEN,
                &format!("{prefix}.attention.self.value"),
            )?;
            let mut context = vec![0.0; n * HIDDEN];
            for head in 0..HEADS {
                let gather = |values: &[f32]| {
                    values
                        .chunks_exact(HIDDEN)
                        .flat_map(|row| row[head * HEAD..(head + 1) * HEAD].iter().copied())
                        .collect::<Vec<_>>()
                };
                let q = gather(&query);
                let k = gather(&key);
                let v = gather(&value);
                let mut scores = matmul(&q, n, HEAD, &k, n, true);
                for row in scores.chunks_exact_mut(n) {
                    for (score, &mask) in row.iter_mut().zip(mask) {
                        *score = (*score as f64 / (HEAD as f64).sqrt()) as f32;
                        *score += if mask == 0 { f32::MIN } else { 0.0 };
                    }
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0;
                    for score in row.iter_mut() {
                        *score = (*score - max).exp();
                        sum += *score;
                    }
                    for score in row {
                        *score /= sum;
                    }
                }
                let head_values = matmul(&scores, n, n, &v, HEAD, false);
                for (row, values) in context
                    .chunks_exact_mut(HIDDEN)
                    .zip(head_values.chunks_exact(HEAD))
                {
                    row[head * HEAD..(head + 1) * HEAD].copy_from_slice(values);
                }
            }
            let mut attention = self.linear(
                &context,
                n,
                HIDDEN,
                HIDDEN,
                &format!("{prefix}.attention.output.dense"),
            )?;
            for (a, h) in attention.iter_mut().zip(&hidden) {
                *a += h;
            }
            self.norm(
                &mut attention,
                &format!("{prefix}.attention.output.LayerNorm"),
            )?;
            let mut intermediate = self.linear(
                &attention,
                n,
                HIDDEN,
                INTERMEDIATE,
                &format!("{prefix}.intermediate.dense"),
            )?;
            for v in &mut intermediate {
                *v = (libm::erff(*v * std::f32::consts::FRAC_1_SQRT_2) + 1.0) * 0.5 * *v;
            }
            hidden = self.linear(
                &intermediate,
                n,
                INTERMEDIATE,
                HIDDEN,
                &format!("{prefix}.output.dense"),
            )?;
            for (h, a) in hidden.iter_mut().zip(attention) {
                *h += a;
            }
            self.norm(&mut hidden, &format!("{prefix}.output.LayerNorm"))?;
        }
        let mut pooled = vec![0.0; HIDDEN];
        for (row, &mask) in hidden.chunks_exact(HIDDEN).zip(mask) {
            for (sum, &v) in pooled.iter_mut().zip(row) {
                *sum += v * mask as f32;
            }
        }
        Ok(pooled)
    }
}

fn matmul(
    lhs: &[f32],
    rows: usize,
    width: usize,
    rhs: &[f32],
    cols: usize,
    transpose: bool,
) -> Vec<f32> {
    assert_eq!(lhs.len(), rows * width);
    assert_eq!(rhs.len(), width * cols);
    let mut output = vec![0.0; rows * cols];
    // All matrices are owned contiguous F32 slices. Lengths are checked above;
    // transposition changes strides only, and output cannot alias either input.
    unsafe {
        gemm::gemm(
            rows,
            cols,
            width,
            output.as_mut_ptr(),
            1,
            cols as isize,
            false,
            lhs.as_ptr(),
            1,
            width as isize,
            rhs.as_ptr(),
            if transpose { width as isize } else { 1 },
            if transpose { 1 } else { cols as isize },
            0.0,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_layouts_match_scalar_products() {
        let a = [1., 2., 3., 4., 5., 6.];
        let b = [7., 8., 9., 10., 11., 12.];
        assert_eq!(matmul(&a, 2, 3, &b, 2, false), [58., 64., 139., 154.]);
        assert_eq!(
            matmul(&a, 2, 3, &[7., 9., 11., 8., 10., 12.], 2, true),
            [58., 64., 139., 154.]
        );
    }

    #[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
    #[test]
    fn f32_backend_matches_candle_coordinates_and_ranking() -> Result<(), Error> {
        let native = Model::load()?;
        let candle = crate::candle::Model::load()?;
        let tokenizer = crate::tokenizer()?;
        let mut native_rows = Vec::new();
        let mut candle_rows = Vec::new();
        let mut worst = 0.0f32;
        for text in [
            "Send a document".to_owned(),
            "Transfer a file".into(),
            "A volcano erupts".into(),
            "The dog chased the cat".into(),
            "The cat chased the dog".into(),
            "package an Obsidian vault into a datapack".into(),
            "".into(),
            "日本語 中文 مرحبا café 🦀".into(),
            "word ".repeat(200),
        ] {
            let encoded = tokenizer.encode(text, true).map_err(inference)?;
            let a = crate::normalize(native.pooled(
                encoded.get_ids(),
                encoded.get_type_ids(),
                encoded.get_attention_mask(),
            )?)?;
            let b = crate::normalize(candle.pooled(
                encoded.get_ids(),
                encoded.get_type_ids(),
                encoded.get_attention_mask(),
            )?)?;
            let max_error = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(max_error < 2e-5, "coordinate drift {max_error}");
            worst = worst.max(max_error);
            native_rows.push(a);
            candle_rows.push(b);
        }
        for query in [
            "I need to transfer a file",
            "an erupting volcano",
            "build an Obsidian datapack",
            "which animal chased the other",
        ] {
            let encoded = tokenizer.encode(query, true).map_err(inference)?;
            let a = crate::normalize(native.pooled(
                encoded.get_ids(),
                encoded.get_type_ids(),
                encoded.get_attention_mask(),
            )?)?;
            let b = crate::normalize(candle.pooled(
                encoded.get_ids(),
                encoded.get_type_ids(),
                encoded.get_attention_mask(),
            )?)?;
            let rank = |query: &[f32], rows: &[Vec<f32>]| {
                let mut scores = rows
                    .iter()
                    .enumerate()
                    .map(|(i, row)| (i, query.iter().zip(row).map(|(a, b)| a * b).sum::<f32>()))
                    .collect::<Vec<_>>();
                scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                scores
                    .into_iter()
                    .take(5)
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>()
            };
            assert_eq!(rank(&a, &native_rows), rank(&b, &candle_rows), "{query}");
        }
        println!(
            "9 Candle/F32 coordinate comparisons: maximum absolute error {worst}; 4 top-five rankings identical"
        );
        Ok(())
    }
}
