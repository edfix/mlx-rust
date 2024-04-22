use std::ops::Range;

use mlx_sys::{mlx_arange, mlx_array_from_bool, mlx_array_from_data, mlx_array_from_float, mlx_array_from_int, mlx_concatenate, mlx_full, mlx_ones, mlx_zeros, mlx_zeros_like};

use crate::{MLXArray, r#type::MlxType, stream::MLXStream, VectorMLXArray};
use crate::stream::get_default_stream;

impl MLXArray {
    /// the data will be copied, so it's safe.
    pub fn array<T: MlxType>(data: &[T], shape: &[i32]) -> Self {
        let handle = unsafe {
            mlx_array_from_data(
                data.as_ptr() as *const ::std::os::raw::c_void,
                shape.as_ptr() as *const ::std::os::raw::c_int,
                shape.len() as ::std::os::raw::c_int,
                T::mlx_array_dtype,
            )
        };
        Self::from_raw(handle)
    }

    // pub fn rand<T: MlxType>(shapes: &[i32]) -> Self {
    //     let handle = unsafe { mlx_array };
    //     Self::from_raw(handle)
    // }

    pub fn arange<T: MlxType>(range: Range<f64>, step: f64, stream: MLXStream) -> Self {
        let handle = unsafe {
            mlx_arange(
                range.start,
                range.end,
                step,
                T::mlx_array_dtype,
                stream.as_ptr(),
            )
        };
        Self::from_raw(handle)
    }

    pub fn cat<T: Into<VectorMLXArray>>(array: T, axis: i32,  stream: MLXStream) -> MLXArray {
        let handle = unsafe {
            mlx_concatenate(array.into().as_ptr(), axis, stream.as_ptr())
        };
        MLXArray::from_raw(handle)
    }

    pub fn ones<T: MlxType>(shape: &[i32], stream: MLXStream) -> MLXArray {
        let handle = unsafe {
            mlx_ones(
                shape.as_ptr() as *const ::std::os::raw::c_int,
                shape.len(),
                T::mlx_array_dtype,
                stream.as_ptr(),
            )
        };
        MLXArray::from_raw(handle)
    }
    pub fn zeros<T: MlxType>(shape: &[i32], stream: MLXStream) -> MLXArray {
        let handle = unsafe {
            mlx_zeros(
                shape.as_ptr() as *const ::std::os::raw::c_int,
                shape.len(),
                T::mlx_array_dtype,
                stream.as_ptr(),
            )
        };
        MLXArray::from_raw(handle)
    }

    pub fn zeros_like(&self, stream: Option<MLXStream>) -> MLXArray {
        let stream = stream.unwrap_or_else(|| get_default_stream());
        let handle = unsafe {
            mlx_zeros_like(
                self.as_ptr(),
                stream.as_ptr(),
            )
        };
        MLXArray::from_raw(handle)
    }

    pub fn full<T: Into<MLXArray>>(shape: &[i32], value: T, stream: MLXStream) -> MLXArray {
        let array: MLXArray = value.into();
        let handle = unsafe {
            mlx_full(
                shape.as_ptr() as *const ::std::os::raw::c_int,
                shape.len(),
                array.as_ptr(),
                array.dtype(),
                stream.as_ptr(),
            )
        };
        MLXArray::from_raw(handle)
    }
}

impl From<f32> for MLXArray {
    fn from(value: f32) -> Self {
        let handle = unsafe { mlx_array_from_float(value) };
        Self::from_raw(handle)
    }
}

impl From<i8> for MLXArray {
    fn from(value: i8) -> Self {
        let handle = unsafe { mlx_array_from_int(value as i32) };
        Self::from_raw(handle)
    }
}

impl From<i16> for MLXArray {
    fn from(value: i16) -> Self {
        let handle = unsafe { mlx_array_from_int(value as i32) };
        Self::from_raw(handle)
    }
}

impl From<i32> for MLXArray {
    fn from(value: i32) -> Self {
        let handle = unsafe { mlx_array_from_int(value) };
        Self::from_raw(handle)
    }
}

impl From<bool> for MLXArray {
    fn from(value: bool) -> Self {
        let handle = unsafe { mlx_array_from_bool(value) };
        Self::from_raw(handle)
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use crate::{array::MLXArray, stream::get_default_stream};

    #[test]
    fn test_array_arange() {
        let r = MLXArray::arange::<f16>(0.0f64..1.0f64, 0.1f64, get_default_stream());
        println!("{}", r)
    }
}
