use mlx_derive::Module;
use mlx_rust::fast::fast_layer_norm;
use mlx_rust::r#type::MlxType;
use mlx_rust::stream::get_default_stream;
use crate::MLXArray;

#[derive(Clone, Debug, Module)]
pub struct LayerNorm {
    weight: Option<MLXArray>,
    bias: Option<MLXArray>,
    #[param(skip)]
    eps: f32,
}

impl LayerNorm {
    pub fn new<T: MlxType>(dim: i32, affine: bool, eps: f32) -> Self {
        let (weight, bias) = if affine {
            (
                Some(MLXArray::ones::<T>(&[dim], get_default_stream())),
                Some(MLXArray::zeros::<T>(&[dim], get_default_stream())),
            )
        } else {
            (None, None)
        };
        Self { weight, bias, eps }
    }
}

impl LayerNorm {
    fn fwd(&self, value: MLXArray) -> MLXArray {
        fast_layer_norm(
            value,
            self.weight.clone(),
            self.bias.clone(),
            self.eps,
            get_default_stream(),
        )
    }
}
