use crate::nn::fast::fast_rms_norm;
use crate::nn::Module;
use crate::r#type::MlxType;
use crate::stream::get_default_stream;
use crate::MLXArray;

pub struct RmsNorm {
    weight: MLXArray,
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
