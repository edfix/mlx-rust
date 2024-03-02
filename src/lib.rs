#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod array;
mod arry_op;
mod as_array;
mod device;
mod stream;
mod string;
mod r#type;
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_create_array() {
        unsafe {
            let a = mlx_array_from_int(2);
            mlx_free(a as *mut c_void);
        }
    }
}
