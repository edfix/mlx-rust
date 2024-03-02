use std::ffi::{c_void, CStr};

use crate::{mlx_free, mlx_string, mlx_string_data};

pub struct MlxString {
    handle: mlx_string,
}

impl MlxString {
    pub fn as_str(&self) -> Result<&str, std::str::Utf8Error> {
        let c_string = unsafe { mlx_string_data(self.handle) };
        // SAFETY: We've checked that the pointer is not null, and we assume that it points to
        // a valid, null-terminated C string. The lifetime of the resulting &str is tied to
        // the lifetime of the C string, so the caller must ensure that the C string is valid
        // for the duration of the &str's use.
        let c_str = unsafe { CStr::from_ptr(c_string) };

        // Convert the CStr to a &str, checking for valid UTF-8
        c_str.to_str()
    }
    pub fn new(handle: mlx_string) -> Self {
        Self { handle }
    }
}

impl Drop for MlxString {
    fn drop(&mut self) {
        unsafe {
            mlx_free(self.handle as *mut c_void);
        }
    }
}
