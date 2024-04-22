use mlx_derive::Module;
use mlx_rust::fast::fast_rms_norm;
use mlx_rust::r#type::MlxType;
use mlx_rust::stream::get_default_stream;
use crate::MLXArray;

#[derive(Clone, Debug, Module)]
pub struct RmsNorm {
    weight: MLXArray,
    #[param(skip)]
    eps: f32,
}

impl RmsNorm {
    pub fn new<T: MlxType>(dim: i32, eps: f32) -> Self {
        let weight = MLXArray::ones::<T>(&[dim], get_default_stream());
        Self { weight, eps }
    }
}
impl RmsNorm {
    fn fwd(&self, value: MLXArray) -> MLXArray {
        fast_rms_norm(value, self.weight.clone(), self.eps, get_default_stream())
    }
}
