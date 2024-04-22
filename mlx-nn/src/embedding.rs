use crate::{
    r#type::MlxType,
    random::{key, uniform},
    stream::get_default_stream,
    MLXArray,
};

use super::Module;

pub struct Embedding {
    w: MLXArray,
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
        Self { w: weight }
    }

    fn fwd(&self, index: MLXArray) -> MLXArray {
        self.w.index_select(0, index, get_default_stream())
    }
}
