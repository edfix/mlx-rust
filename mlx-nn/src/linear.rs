use crate::{
    array_op::addmm,
    r#type::MlxType,
    random::{key, uniform},
    stream::get_default_stream,
    MLXArray,
};

use super::Module;

pub struct Linear {
    w: MLXArray,
    bias: Option<MLXArray>,
}

impl Linear {
    pub fn new<T: MlxType>(in_features: usize, out_features: usize, has_bias: bool) -> Self {
        let scale = f32::sqrt(1.0 / in_features as f32);
        let rng_key = key(1);
        let weight = uniform::<T>(
            -scale..=scale,
            &[out_features as i32, in_features as i32],
            rng_key,
            get_default_stream(),
        );
        let rng_key = key(1);
        let bias = if has_bias {
            let bias = uniform::<T>(
                -scale..=scale,
                &[out_features as i32],
                rng_key,
                get_default_stream(),
            );
            Some(bias)
        } else {
            None
        };
        Self {
            w: weight,
            bias: bias,
        }
    }
}

impl Linear {
    fn fwd(&self, value: MLXArray) -> MLXArray {
        match &self.bias {
            None => value.matmul(self.w.t(), get_default_stream()),
            Some(bias) => addmm(
                bias.clone(),
                value,
                self.w.t(),
                1.0,
                1.0,
                get_default_stream(),
            ),
        }
    }
}
