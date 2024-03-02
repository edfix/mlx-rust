use std::ffi::c_void;
use std::fmt::{Display, Formatter};
use std::rc::Rc;
use std::slice;

use crate::r#type::MlxType;
use crate::string::MlxString;
use crate::{
    mlx_array, mlx_array_from_data, mlx_array_itemsize, mlx_array_nbytes, mlx_array_ndim,
    mlx_array_shape, mlx_array_size, mlx_array_strides, mlx_free, mlx_tostring,
};

#[derive(Clone)]
pub struct MlxArray {
    handle: Rc<InnerMlxArray>,
}

struct InnerMlxArray(mlx_array);
impl Drop for InnerMlxArray {
    fn drop(&mut self) {
        unsafe {
            mlx_free(self.0 as *mut c_void);
        }
    }
}

impl MlxArray {
    #[inline]
    pub(crate) fn from_raw(handle: mlx_array) -> Self {
        Self {
            handle: Rc::new(InnerMlxArray(handle)),
        }
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> mlx_array {
        self.handle.as_ref().0
    }

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
}

impl MlxArray {
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
            slice::from_raw_parts_mut(p, len)
        }
    }
}

impl Display for MlxArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mlxstr = unsafe { mlx_tostring(self.as_ptr() as *mut c_void) };
        let bind_str = MlxString::new(mlxstr);
        f.write_str(bind_str.as_str().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use crate::array::MlxArray;

    #[test]
    fn it_works() {
        let array: MlxArray = 12.0.into();
        let array1 = MlxArray::array(&[123., 134.], &[2]);
        println!("{}", array + array1.clone() + array1)
    }
}
