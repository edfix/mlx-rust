use mlx_sys::{mlx_compile, mlx_disable_compile, mlx_enable_compile};

use crate::{
    closure::{MLXClosure, MLXFunc},
    VectorMLXArray,
};

pub fn compile<F, IN, OUT>(f: F, shapeless: bool) -> MLXClosure<IN, OUT>
where
    F: MLXFunc<IN, OUT>,
    IN: for<'a> From<&'a VectorMLXArray> + Into<VectorMLXArray> + 'static,
    OUT: for<'b> From<&'b VectorMLXArray> + Into<VectorMLXArray> + 'static,
{
    let closure = MLXClosure::new(f);
    let handle = unsafe { mlx_compile(closure.as_ptr(), shapeless) };
    MLXClosure::from_raw(handle)
}

pub fn enable_compile() {
    unsafe { mlx_enable_compile() }
}

pub fn disable_compile() {
    unsafe { mlx_disable_compile() }
}
