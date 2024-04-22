use std::f32::consts::PI;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use serde::Deserialize;

use mlx_rust::array_op::{erf, sigmoid, sqrt, square, tanh};
use mlx_rust::closure::MLXFunc;
use mlx_rust::compile::compile;

use crate::activations::Activation::gelu_new;
use crate::MLXArray;

#[derive(PartialEq, Deserialize, Clone, Debug)]
pub enum Activation {
    Gelu,
    gelu_new,
    Relu,
    // Relu2,
    // Relu6,
    // Silu,
}

fn gelu(x: MLXArray) -> MLXArray {
    x.clone() * (1 + erf(x / sqrt(2.0.into()))) / 2
}

fn new_gelu(x: MLXArray) -> MLXArray {
    0.5 * x.clone() * (1.0 + tanh((2.0 / PI).sqrt() * (x.clone() + 0.044715 * x.clone().powf(&3.0.into(), None))))
}

pub fn approximate_gelu(x: MLXArray) -> MLXArray {
    &x * sigmoid(1.60033 * &x * (1 + 0.0433603 * square(x.clone())))
}

// static compiled_gelu: Box<dyn Fn(MLXArray) -> MLXArray>  = {
//     Box::new(compile(gelu, true))
// };
//
//
// static compiled_new_gelu: Box<dyn Fn(MLXArray) -> MLXArray>  = {
//     Box::new(compile(new_gelu, true))
// };

#[derive(Clone)]
pub struct ActivationLayer {
    f: Arc<Box<dyn Fn(MLXArray) -> MLXArray>>
}

impl Debug for ActivationLayer {
    fn fmt(&self, _: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl  ActivationLayer {
    pub fn new(activation: Activation) -> Self {
        let f = match activation {
            Activation::Gelu => {
               let f: Box<dyn Fn(MLXArray) -> MLXArray> =  Box::new(compile(gelu, true));
                f
            }
            gelu_new => {
                let f: Box<dyn Fn(MLXArray) -> MLXArray> = Box::new(compile(new_gelu, true));
                f
            }
            Activation::Relu => {
                let f: Box<dyn Fn(MLXArray) -> MLXArray> = Box::new(|x: MLXArray| x.maximum_with_stream(&(0.into()), None));
                f
            }
        };
        Self { f: Arc::new(f) }
    }

    pub fn fwd(&self, value: MLXArray) -> MLXArray {
        self.f.apply(value)
    }
}
