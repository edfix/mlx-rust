use std::ptr;

use mlx_sys::{mlx_fast_layer_norm, mlx_fast_scaled_dot_product_attention};
use mlx_sys::mlx_fast_rms_norm;
use mlx_sys::mlx_fast_rope;

use crate::{MLXArray, stream::MLXStream};

pub fn fast_RoPE(
    array: MLXArray,
    dim: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: i32,
    stream: MLXStream,
) -> MLXArray {
    let handle = unsafe {
        mlx_fast_rope(
            array.as_ptr(),
            dim,
            traditional,
            base,
            scale,
            offset,
            stream.as_ptr(),
        )
    };
    MLXArray::from_raw(handle)
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
///
pub fn fast_scaled_dot_product_attention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: f32,
    mask: Option<MLXArray>,
    stream: MLXStream,
) -> MLXArray {
    let handle = unsafe {
        mlx_fast_scaled_dot_product_attention(
            queries.as_ptr(),
            keys.as_ptr(),
            values.as_ptr(),
            scale,
            mask.map(|b|b.as_ptr()).unwrap_or(ptr::null_mut()),
            stream.as_ptr(),
        )
    };
    MLXArray::from_raw(handle)
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
pub fn fast_layer_norm(x: MLXArray, weight: Option<MLXArray>, bias: Option<MLXArray>, eps: f32, stream: MLXStream) -> MLXArray {
    let handle = unsafe {
        mlx_fast_layer_norm(
            x.as_ptr(),
            weight.map(|b|b.as_ptr()).unwrap_or(ptr::null_mut()),
            bias.map(|b|b.as_ptr()).unwrap_or(ptr::null_mut()),
            eps,
            stream.as_ptr(),
        )
    };
    MLXArray::from_raw(handle)
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
pub fn fast_rms_norm(x: MLXArray, weight: MLXArray, eps: f32, stream: MLXStream) -> MLXArray {
    let handle = unsafe { mlx_fast_rms_norm(x.as_ptr(), weight.as_ptr(), eps, stream.as_ptr()) };
    MLXArray::from_raw(handle)

}
