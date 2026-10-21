use crate::{Error, inference, weights};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};

pub(crate) struct Model(BertModel);

impl Model {
    pub fn load() -> Result<Self, Error> {
        let tensors = weights::load()?
            .into_iter()
            .map(|(name, weight)| {
                Tensor::from_vec(weight.data, weight.shape, &Device::Cpu)
                    .map(|tensor| (name, tensor))
                    .map_err(inference)
            })
            .collect::<Result<_, _>>()?;
        let config: Config =
            serde_json::from_str(include_str!("../assets/config.json")).map_err(inference)?;
        BertModel::load(
            VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
            &config,
        )
        .map(Self)
        .map_err(inference)
    }

    pub fn pooled(&self, ids: &[u32], types: &[u32], mask: &[u32]) -> Result<Vec<f32>, Error> {
        self.forward(ids, types, mask).map_err(inference)
    }

    fn forward(&self, ids: &[u32], types: &[u32], mask: &[u32]) -> candle_core::Result<Vec<f32>> {
        let ids = Tensor::new(ids, &Device::Cpu)?.unsqueeze(0)?;
        let types = Tensor::new(types, &Device::Cpu)?.unsqueeze(0)?;
        let mask = Tensor::new(mask, &Device::Cpu)?.unsqueeze(0)?;
        let hidden = self.0.forward(&ids, &types, Some(&mask))?;
        hidden
            .broadcast_mul(&mask.to_dtype(DType::F32)?.unsqueeze(2)?)?
            .sum(1)?
            .squeeze(0)?
            .to_vec1::<f32>()
    }
}
