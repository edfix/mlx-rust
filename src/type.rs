use crate::{
    mlx_array_dtype_, mlx_array_dtype__MLX_BFLOAT16, mlx_array_dtype__MLX_BOOL,
    mlx_array_dtype__MLX_FLOAT16, mlx_array_dtype__MLX_FLOAT32, mlx_array_dtype__MLX_INT16,
    mlx_array_dtype__MLX_INT32, mlx_array_dtype__MLX_INT64, mlx_array_dtype__MLX_INT8,
    mlx_array_dtype__MLX_UINT16, mlx_array_dtype__MLX_UINT32, mlx_array_dtype__MLX_UINT64,
    mlx_array_dtype__MLX_UINT8,
};
use half::{bf16, f16};

pub trait MlxType {
    const mlx_array_dtype: mlx_array_dtype_;
}

macro_rules! impl_mlx_type {
    ($type:ty, $dtype:expr) => {
        impl MlxType for $type {
            const mlx_array_dtype: mlx_array_dtype_ = $dtype;
        }
    };
}

// Use the macro to implement MlxType for bool, i32, i16, i8
impl_mlx_type!(bool, mlx_array_dtype__MLX_BOOL);
impl_mlx_type!(u8, mlx_array_dtype__MLX_UINT8);
impl_mlx_type!(u16, mlx_array_dtype__MLX_UINT16);
impl_mlx_type!(u32, mlx_array_dtype__MLX_UINT32);
impl_mlx_type!(u64, mlx_array_dtype__MLX_UINT64);
impl_mlx_type!(i8, mlx_array_dtype__MLX_INT8);
impl_mlx_type!(i16, mlx_array_dtype__MLX_INT16);
impl_mlx_type!(i32, mlx_array_dtype__MLX_INT32);
impl_mlx_type!(i64, mlx_array_dtype__MLX_INT64);
impl_mlx_type!(f16, mlx_array_dtype__MLX_FLOAT16);
impl_mlx_type!(f32, mlx_array_dtype__MLX_FLOAT32);
impl_mlx_type!(bf16, mlx_array_dtype__MLX_BFLOAT16);
