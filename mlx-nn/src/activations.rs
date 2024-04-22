use crate::nn::Module;
use crate::stream::get_default_stream;
use crate::MLXArray;

pub enum Activation {
    // Gelu,
    // NewGelu,
    Relu,
    // Relu2,
    // Relu6,
    // Silu,
}

impl Activation {
    fn fwd(&self, value: MLXArray) -> MLXArray {
        match self {
            // Activation::Gelu => {}
            // Activation::NewGelu => {}
            Activation::Relu => value.maximum_with_stream(&(0.into()), get_default_stream()), // Activation::Relu2 => {}
        }
    }
}
