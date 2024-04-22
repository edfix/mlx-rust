use crate::nn::activations::Activation;
use crate::nn::linear::Linear;
use crate::nn::Module;
use crate::r#type::MlxType;
use crate::MLXArray;

struct MLP {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
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
        Self { fc1, fc2, act }
    }
}

impl MLP {
    fn forward(&self, x: MLXArray) -> MLXArray {
        // let y = self.fc1.forward(x);
        // let y = self.act.forward(y);
        // self.fc2.forward(y)
        todo!()
    }
}
