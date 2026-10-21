use crate::{Error, inference};
use std::collections::HashMap;

pub(crate) struct Weight {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub(crate) fn load() -> Result<HashMap<String, Weight>, Error> {
    let file = safetensors::SafeTensors::deserialize(include_bytes!("../assets/model.safetensors"))
        .map_err(inference)?;
    let shapes: HashMap<String, Vec<usize>> =
        serde_json::from_str(include_str!("../assets/shapes.json")).map_err(inference)?;
    let mut weights = HashMap::new();
    for (name, shape) in shapes {
        let count = shape.iter().product::<usize>();
        let data = if shape.len() < 2 {
            file.tensor(&name)
                .map_err(inference)?
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>()
        } else {
            let packed = file.tensor(&format!("{name}.q4")).map_err(inference)?;
            let scales = file.tensor(&format!("{name}.scales")).map_err(inference)?;
            let scales = scales
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::f16::from_bits(u16::from_le_bytes(*b)).to_f32())
                .collect::<Vec<_>>();
            if packed.data().len() * 2 < count || scales.len() * 32 < count {
                return Err(inference("invalid bundled quantized tensor length"));
            }
            (0..count)
                .map(|i| {
                    let nibble = (packed.data()[i / 2] >> ((i % 2) * 4)) & 15;
                    (nibble as i16 - 8) as f32 * scales[i / 32]
                })
                .collect()
        };
        if data.len() != count {
            return Err(inference("invalid bundled tensor length"));
        }
        weights.insert(name, Weight { shape, data });
    }
    Ok(weights)
}
