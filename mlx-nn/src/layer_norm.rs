use crate::nn::fast::fast_layer_norm;
use crate::nn::Module;
use crate::r#type::MlxType;
use crate::stream::get_default_stream;
use crate::MLXArray;

pub struct LayerNorm {
    weight: Option<MLXArray>,
    bias: Option<MLXArray>,
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
