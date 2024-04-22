use mlx_derive::Module;
use mlx_rust::array_op::addmm;
use mlx_rust::r#type::MlxType;
use mlx_rust::random::{key, uniform};
use mlx_rust::stream::get_default_stream;

use crate::MLXArray;

#[derive(Clone, Debug, Module)]
pub struct Linear {
    weight: MLXArray,
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
            // let bias = MLXArray::ones::<T>(&[out_features as i32], get_default_stream());
            Some(bias)
        } else {
            None
        };
        Self {
            weight: weight,
            bias: bias,
        }
    }
}

impl Linear {
    fn fwd(&self, value: MLXArray) -> MLXArray {
        match &self.bias {
            None => value.matmul(self.weight.t(None), None),
            Some(bias) => addmm(
                &value,
                bias,
                &self.weight,
                1.0,
                1.0,
                None
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use mlx_rust::MLXArray;
    use mlx_rust::module::Module;
    use mlx_rust::stream::get_default_stream;

    use crate::linear::Linear;

    #[test]
    pub fn test_forward_params() {
        let x = MLXArray::ones::<f32>(&[3, 1024], get_default_stream());
        let linear = Linear::new::<f32>(1024, 1024, true);
        let y = linear.forward(x);
        println!("{}", y)
    }

    #[test]
    pub fn test_update_params() {
        let in_features = 1024;
        let out_features = 1024;
        let x = MLXArray::ones::<f32>(&[3, in_features as i32], get_default_stream());
        let mut linear = Linear::new::<f32>(in_features, out_features, true);
        let mut params: HashMap<String, MLXArray> = HashMap::new();
        params.insert("weight".into(), MLXArray::ones::<f32>(&[out_features as i32, in_features as i32], get_default_stream()));
        params.insert("bias".into(), MLXArray::ones::<f32>(&[out_features as i32], get_default_stream()));
        linear.update_named_params("", &mut params);
        let y = linear.forward(x);
        println!("{}", y);
        // assert_eq!(y, MLXArray::ones::<f32>(&[out_features as i32], get_default_stream()))
    }
}
