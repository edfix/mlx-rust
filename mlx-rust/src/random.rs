use std::ops::RangeInclusive;

use mlx_sys::{mlx_random_categorical, mlx_random_categorical_shape, mlx_random_key, mlx_random_normal, mlx_random_randint, mlx_random_split_equal_parts, mlx_random_uniform};

use crate::{MLXArray, r#type::MlxType, stream::MLXStream};
use crate::stream::get_default_stream;

pub fn key(seed: u64) -> MLXArray {
    let handle = unsafe { mlx_random_key(seed) };
    MLXArray::from_raw(handle)
}

pub fn split(key: MLXArray, num: usize, stream: MLXStream) -> MLXArray {
    let handle = unsafe {
        mlx_random_split_equal_parts(key.as_ptr(), num as ::std::os::raw::c_int, stream.as_ptr())
    };
    MLXArray::from_raw(handle)
}

// pub fn split_two(key: MLXArray, stream: MLXStream) -> (MLXArray, MLXArray) {
//     let r = split(key, 2, stream);
//     (r.get(0).unwrap(), r.get(1).unwrap())
// }

pub fn uniform<T: MlxType>(
    range: RangeInclusive<f32>,
    shape: &[i32],
    key: MLXArray,
    stream: MLXStream,
) -> MLXArray {
    let lb: MLXArray = range.start().clone().into();
    let ub: MLXArray = range.end().clone().into();
    let handle = unsafe {
        mlx_random_uniform(
            lb.as_ptr(),
            ub.as_ptr(),
            shape.as_ptr() as *const ::std::os::raw::c_int,
            shape.len(),
            T::mlx_array_dtype,
            key.as_ptr(),
            stream.as_ptr(),
        )
    };
    MLXArray::from_raw(handle)
}

pub fn normal<T: MlxType>(
    shape: &[i32],
    mean: f32,
    std: f32,
    key: MLXArray,
    stream: MLXStream,
) -> MLXArray {
    let handle = unsafe {
        mlx_random_normal(
            shape.as_ptr() as *const ::std::os::raw::c_int,
            shape.len(),
            T::mlx_array_dtype,
            mean,
            std,
            key.as_ptr(),
            stream.as_ptr(),
        )
    };
    MLXArray::from_raw(handle)
}

pub fn randint<T: MlxType>(
    range: RangeInclusive<i32>,
    shape: &[i32],
    key: MLXArray,
    stream: MLXStream,
) -> MLXArray {
    let lb: MLXArray = range.start().clone().into();
    let ub: MLXArray = range.end().clone().into();
    let handle = unsafe {
        mlx_random_randint(
            lb.as_ptr(),
            ub.as_ptr(),
            shape.as_ptr() as *const ::std::os::raw::c_int,
            shape.len(),
            T::mlx_array_dtype,
            key.as_ptr(),
            stream.as_ptr(),
        )
    };
    MLXArray::from_raw(handle)
}

pub fn categorical(
    logits: MLXArray,
    axis: i32,
    shape:  Option<&[i32]>,
    key0: Option<MLXArray>,
    stream: Option<MLXStream>,
) -> MLXArray {
    let key = key0.unwrap_or_else(|| key(0));
    let stream = stream.unwrap_or_else(||get_default_stream());
    let handle = if let Some(shape) =  shape {
        unsafe {
            mlx_random_categorical_shape(
                logits.as_ptr(),
                axis,
                shape.as_ptr() as *const ::std::os::raw::c_int,
                shape.len(),
                key.as_ptr(),
                stream.as_ptr()
            )
        }
    } else {
        unsafe { mlx_random_categorical(logits.as_ptr(), axis, key.as_ptr(), stream.as_ptr()) }
    };
    MLXArray::from_raw(handle)
}
