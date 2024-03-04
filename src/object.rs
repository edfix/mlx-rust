use std::{
    ffi::c_void,
    fmt::{Debug, Display, Formatter},
    marker::PhantomData,
};

use crate::{mlx_free, mlx_retain, mlx_tostring, string::MLXString};

///MLXObject is a reference-counted smart pointer that provides a zero-overhead abstraction over the mlx-c mlx_object.
///It functions similarly to Rust's Rc type,
//managing the lifetime of the underlying mlx_object through reference counting.
pub(crate) struct MLXObject<T> {
    ptr: *mut T,
    data: PhantomData<T>,
}

impl<T> Clone for MLXObject<T> {
    fn clone(&self) -> Self {
        unsafe {
            mlx_retain((self.ptr) as *mut c_void);
        }
        Self {
            ptr: self.ptr,
            data: PhantomData::default(),
        }
    }
}

impl<T> Drop for MLXObject<T> {
    fn drop(&mut self) {
        unsafe {
            mlx_free((self.ptr) as *mut c_void);
        }
    }
}

impl<T> MLXObject<T> {
    pub(crate) fn from_raw(handler: *mut T) -> Self {
        Self {
            ptr: handler,
            data: PhantomData::default(),
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut T {
        self.ptr
    }
}

impl<T> Debug for MLXObject<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{:?}", self.as_ptr()))
    }
}

impl<T> Display for MLXObject<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mlxstr = unsafe { mlx_tostring(self.as_ptr() as *mut c_void) };
        let bind_str = MLXString::new(mlxstr);
        f.write_str(bind_str.as_str().unwrap())
    }
}

impl<T> PartialEq for MLXObject<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

unsafe impl<T> Send for MLXObject<T> {}
