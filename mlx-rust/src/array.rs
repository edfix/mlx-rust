use std::fmt::{Display, Formatter};
use std::slice;

use mlx_sys::{mlx_array, mlx_array_, mlx_array_dim, mlx_array_dtype_, mlx_array_eval, mlx_array_get_dtype, mlx_array_itemsize, mlx_array_nbytes, mlx_array_ndim, mlx_array_shape, mlx_array_size, mlx_array_strides, mlx_astype, mlx_expand_dims, mlx_reshape, mlx_transpose};

use crate::object::MLXObject;
use crate::r#type::MlxType;
use crate::stream::get_default_stream;

#[derive(Clone, Debug, PartialEq)]
pub struct MLXArray {
    handle: MLXObject<mlx_array_>,
}

impl MLXArray {
    #[inline]
    pub fn from_raw(handle: mlx_array) -> Self {
        Self {
            handle: MLXObject::from_raw(handle),
        }
    }

    #[inline]
    pub fn as_ptr(&self) -> mlx_array {
        self.handle.as_ptr()
    }

    pub(crate) fn dtype(&self) -> mlx_array_dtype_ {
        unsafe { mlx_array_get_dtype(self.as_ptr()) }
    }
}

impl MLXArray {
    /// Number of elements in the array.
    ///
    #[inline]
    pub fn size(&self) -> usize {
        unsafe { mlx_array_size(self.as_ptr()) }
    }

    /// The size of the array's datatype in bytes.
    #[inline]
    pub fn item_size(&self) -> usize {
        unsafe { mlx_array_itemsize(self.as_ptr()) }
    }

    /// The number of bytes in the array.
    #[inline]
    pub fn nbytes(&self) -> usize {
        unsafe { mlx_array_nbytes(self.as_ptr()) }
    }

    ///The strides of the array.
    pub fn strides(&self) -> &[usize] {
        unsafe {
            let arr = self.as_ptr();
            let len = mlx_array_ndim(arr);
            let p = mlx_array_strides(arr);
            slice::from_raw_parts_mut(p, len)
        }
    }

    ///The shape of the array
    pub fn shape(&self) -> &[i32] {
        unsafe {
            let arr = self.as_ptr();
            let len = mlx_array_ndim(arr);
            let p = mlx_array_shape(arr);
            slice::from_raw_parts(p, len)
        }
    }

    pub fn reshape(&self, shape: &[i32]) -> MLXArray {
        let handle = unsafe {
            mlx_reshape(
                self.as_ptr(),
                shape.as_ptr() as *const ::std::os::raw::c_int,
                shape.len(),
                get_default_stream().as_ptr(),
            )
        };
        MLXArray::from_raw(handle)
    }

    pub fn dim(&self, dim: i32) -> i32 {
        unsafe {
            mlx_array_dim(self.as_ptr(), dim)
        }
    }

    pub fn expand_dims(&self, axes: &[i32]) -> MLXArray {
        let handle = unsafe {
            mlx_expand_dims(
                self.as_ptr(),
                axes.as_ptr() as *const ::std::os::raw::c_int,
                axes.len(),
                get_default_stream().as_ptr()
            )
        };
        MLXArray::from_raw(handle)
    }

    pub fn transpose(&self, dims: &[i32]) -> MLXArray {
        let handle = unsafe {
            mlx_transpose(
                self.as_ptr(),
                dims.as_ptr() as *const ::std::os::raw::c_int,
                dims.len(),
                get_default_stream().as_ptr()
            )
        };
        MLXArray::from_raw(handle)
    }

    pub fn as_type<T: MlxType>(&self) -> MLXArray {
        let handle = unsafe  {
            mlx_astype(self.as_ptr(), T::mlx_array_dtype, get_default_stream().as_ptr())
        };
        MLXArray::from_raw(handle)
    }
    pub fn eval(&self) {
        unsafe { mlx_array_eval(self.as_ptr()) }
    }
}

impl Display for MLXArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.handle.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use crate::array::MLXArray;

    #[test]
    fn it_works() {
        let array: MLXArray = 12.0.into();
        let array1 = MLXArray::array(&[123., 134.], &[2]);
        println!("{}", array + array1.clone() + array1)
    }
}
