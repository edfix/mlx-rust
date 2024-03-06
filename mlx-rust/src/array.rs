use std::fmt::{Display, Formatter};
use std::slice;

use crate::object::MLXObject;
use mlx_sys::{
    mlx_array, mlx_array_, mlx_array_dtype_, mlx_array_eval, mlx_array_get_dtype,
    mlx_array_itemsize, mlx_array_nbytes, mlx_array_ndim, mlx_array_shape, mlx_array_size,
    mlx_array_strides,
};

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
    pub fn shapes(&self) -> &[i32] {
        unsafe {
            let arr = self.as_ptr();
            let len = mlx_array_ndim(arr);
            let p = mlx_array_shape(arr);
            slice::from_raw_parts(p, len)
        }
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
