use std::time::Instant;
use half::f16;
use mlx_derive::Module;
use mlx_rust::module::Module;
use mlx_rust::r#type::MlxType;
use crate::activations::{Activation, ActivationLayer};
use crate::linear::Linear;
use crate::MLXArray;

#[derive(Clone, Debug, Module)]
pub struct MLP {
    fc1: Linear,
    fc2: Linear,
    #[param(skip)]
    act: ActivationLayer,
}

impl MLP {
    pub fn new<T: MlxType>(
        hidden_size: usize,
        intermediate_size: usize,
        bias: bool,
        act: Activation,
    ) -> Self {
        let fc1 = Linear::new::<T>(hidden_size, intermediate_size, bias);
        let fc2 = Linear::new::<T>(intermediate_size, hidden_size, bias);
        let act = ActivationLayer::new(act);
        Self { fc1, fc2, act }
    }
}

impl MLP {
    fn fwd(&self, x: MLXArray) -> MLXArray {
        // x.eval();
        // let instant = Instant::now();
        // println!("mlp x: {}", x);
        let y = self.fc1.forward(x);
        let y = self.act.fwd(y);
        // y.eval();
        let y = y.as_type::<f16>();
        // println!("act time: {:?}", instant.elapsed());

        self.fc2.forward(y)
    }
}
