use std::ffi::{CStr, CString};

use mlx_sys::{mlx_string, mlx_string_, mlx_string_data, mlx_string_new};

use crate::object::MLXObject;

pub struct MLXString {
    handle: MLXObject<mlx_string_>,
}

impl MLXString {
    pub(crate) fn as_str(&self) -> Result<&str, std::str::Utf8Error> {
        let c_string = unsafe { mlx_string_data(self.handle.as_ptr()) };
        // SAFETY: We've checked that the pointer is not null, and we assume that it points to
        // a valid, null-terminated C string. The lifetime of the resulting &str is tied to
        // the lifetime of the C string, so the caller must ensure that the C string is valid
        // for the duration of the &str's use.
        let c_str = unsafe { CStr::from_ptr(c_string) };

        // Convert the CStr to a &str, checking for valid UTF-8
        c_str.to_str()
    }

    pub(crate) fn to_string(&self) -> Result<String, std::str::Utf8Error> {
        let c_string = unsafe { mlx_string_data(self.handle.as_ptr()) };
        // SAFETY: We've checked that the pointer is not null, and we assume that it points to
        // a valid, null-terminated C string. The lifetime of the resulting &str is tied to
        // the lifetime of the C string, so the caller must ensure that the C string is valid
        // for the duration of the &str's use.
        let c_str = unsafe { CStr::from_ptr(c_string) };

        // Convert the CStr to a &str, checking for valid UTF-8
        Ok(c_str.to_string_lossy().into_owned())
    }

    pub(crate) fn from_raw(handle: mlx_string) -> Self {
        Self {
            handle: MLXObject::from_raw(handle),
        }
    }

    pub fn as_ptr(&self) -> mlx_string {
        self.handle.as_ptr()
    }

    pub fn new(value: &str) -> Self {
        let handle = unsafe { mlx_string_new(CString::new(value).unwrap().as_ptr()) };
        Self::from_raw(handle)
    }
}


#[cfg(test)]
mod tests {
    use crate::string::MLXString;

    #[test]
    pub fn test_create_string() {
        let a = MLXString::new("111");
        assert_eq!("111", a.to_string().unwrap());
    }
}