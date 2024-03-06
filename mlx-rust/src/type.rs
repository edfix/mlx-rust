use half::{bf16, f16};
use mlx_sys::{
    mlx_array, mlx_array_data_bool, mlx_array_data_float16, mlx_array_data_float32,
    mlx_array_data_int16, mlx_array_data_int32, mlx_array_data_int64, mlx_array_data_int8,
    mlx_array_data_uint16, mlx_array_data_uint32, mlx_array_data_uint64, mlx_array_data_uint8,
    mlx_array_dtype_, mlx_array_dtype__MLX_BFLOAT16, mlx_array_dtype__MLX_BOOL,
    mlx_array_dtype__MLX_FLOAT16, mlx_array_dtype__MLX_FLOAT32, mlx_array_dtype__MLX_INT16,
    mlx_array_dtype__MLX_INT32, mlx_array_dtype__MLX_INT64, mlx_array_dtype__MLX_INT8,
    mlx_array_dtype__MLX_UINT16, mlx_array_dtype__MLX_UINT32, mlx_array_dtype__MLX_UINT64,
    mlx_array_dtype__MLX_UINT8, mlx_array_item_bfloat16, mlx_array_item_bool,
    mlx_array_item_float16, mlx_array_item_float32, mlx_array_item_int16, mlx_array_item_int32,
    mlx_array_item_int64, mlx_array_item_int8, mlx_array_item_uint16, mlx_array_item_uint32,
    mlx_array_item_uint64, mlx_array_item_uint8,
};

pub trait MlxType {
    const mlx_array_dtype: mlx_array_dtype_;
}

pub trait ScalarMlxType: MlxType {
    const mlx_array_dtype: mlx_array_dtype_;

    unsafe fn to_scalar(handle: mlx_array) -> Self;
    unsafe fn to_slice(handle: mlx_array) -> *const Self;
}

impl<T: ScalarMlxType> MlxType for T {
    const mlx_array_dtype: mlx_array_dtype_ = <T as ScalarMlxType>::mlx_array_dtype;
}

macro_rules! impl_mlx_type {
    ($type:ty, $dtype:expr, $scalar_op: ident, $array_op: ident) => {
        impl ScalarMlxType for $type {
            const mlx_array_dtype: mlx_array_dtype_ = $dtype;

            unsafe fn to_scalar(handle: mlx_array) -> Self {
                $scalar_op(handle)
            }

            unsafe fn to_slice(handle: mlx_array) -> *const Self {
                $array_op(handle)
            }
        }
    };
}

// Use the macro to implement MlxType for bool, i32, i16, i8
impl_mlx_type! {bool, mlx_array_dtype__MLX_BOOL, mlx_array_item_bool, mlx_array_data_bool}
impl_mlx_type! {u8, mlx_array_dtype__MLX_UINT8, mlx_array_item_uint8, mlx_array_data_uint8}
impl_mlx_type! {u16, mlx_array_dtype__MLX_UINT16, mlx_array_item_uint16, mlx_array_data_uint16}
impl_mlx_type! {u32, mlx_array_dtype__MLX_UINT32, mlx_array_item_uint32, mlx_array_data_uint32}
impl_mlx_type! {u64, mlx_array_dtype__MLX_UINT64, mlx_array_item_uint64, mlx_array_data_uint64}
impl_mlx_type! {i8, mlx_array_dtype__MLX_INT8, mlx_array_item_int8, mlx_array_data_int8}
impl_mlx_type! {i16, mlx_array_dtype__MLX_INT16, mlx_array_item_int16, mlx_array_data_int16}
impl_mlx_type! {i32, mlx_array_dtype__MLX_INT32, mlx_array_item_int32, mlx_array_data_int32}
impl_mlx_type! {i64, mlx_array_dtype__MLX_INT64, mlx_array_item_int64, mlx_array_data_int64}
impl_mlx_type! {f32, mlx_array_dtype__MLX_FLOAT32, mlx_array_item_float32, mlx_array_data_float32}
impl ScalarMlxType for f16 {
    const mlx_array_dtype: mlx_array_dtype_ = mlx_array_dtype__MLX_FLOAT16;

    unsafe fn to_scalar(handle: mlx_array) -> Self {
        let r = mlx_array_item_float16(handle);
        f16::from_bits(r)
    }

    unsafe fn to_slice(handle: mlx_array) -> *const Self {
        let r = mlx_array_data_float16(handle);
        r as *const f16
    }
}

impl ScalarMlxType for bf16 {
    const mlx_array_dtype: mlx_array_dtype_ = mlx_array_dtype__MLX_BFLOAT16;

    unsafe fn to_scalar(handle: mlx_array) -> Self {
        let r = mlx_array_item_bfloat16(handle);
        bf16::from_bits(r)
    }

    unsafe fn to_slice(handle: mlx_array) -> *const Self {
        let r = mlx_array_data_float16(handle);
        r as *const bf16
    }
}
