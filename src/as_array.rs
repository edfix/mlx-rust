use crate::{array::MlxArray, mlx_array_from_bool, mlx_array_from_float, mlx_array_from_int};

impl From<f32> for MlxArray {
    fn from(value: f32) -> Self {
        let handle = unsafe { mlx_array_from_float(value) };
        Self::from_raw(handle)
    }
}

impl From<i8> for MlxArray {
    fn from(value: i8) -> Self {
        let handle = unsafe { mlx_array_from_int(value as i32) };
        Self::from_raw(handle)
    }
}

impl From<i16> for MlxArray {
    fn from(value: i16) -> Self {
        let handle = unsafe { mlx_array_from_int(value as i32) };
        Self::from_raw(handle)
    }
}

impl From<i32> for MlxArray {
    fn from(value: i32) -> Self {
        let handle = unsafe { mlx_array_from_int(value) };
        Self::from_raw(handle)
    }
}

impl From<bool> for MlxArray {
    fn from(value: bool) -> Self {
        let handle = unsafe { mlx_array_from_bool(value) };
        Self::from_raw(handle)
    }
}
