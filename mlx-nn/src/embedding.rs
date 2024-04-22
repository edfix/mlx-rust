use mlx_derive::Module;
use mlx_rust::r#type::MlxType;
use mlx_rust::random::{key, uniform};
use mlx_rust::stream::get_default_stream;

use crate::MLXArray;

#[derive(Clone, Debug, Module)]
pub struct Embedding {
    weight: MLXArray,
}

impl Embedding {
    pub fn new<T: MlxType>(num_embeddings: usize, features: usize) -> Self {
        let scale = f32::sqrt(1.0 / features as f32);
        let rng_key = key(1);
        let weight = uniform::<T>(
            -scale..=scale,
            &[num_embeddings as i32, features as i32],
            rng_key,
            get_default_stream(),
        );
        Self { weight: weight }
    }

    fn fwd(&self, index: MLXArray) -> MLXArray {
        self.weight.index_select(0, index, None)
    }
}
