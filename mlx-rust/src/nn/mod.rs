use crate::array_op::addmm;
use crate::stream::get_default_stream;
use crate::MLXArray;

trait Module {
    fn forward(&self, value: MLXArray) -> MLXArray;
}

struct Linear {
    w: MLXArray,
    bias: Option<MLXArray>,
}

impl Module for Linear {
    fn forward(&self, value: MLXArray) -> MLXArray {
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
