use std::marker::PhantomData;

use mlx_sys::{
    mlx_detail_compile, mlx_detail_compile_erase, mlx_disable_compile,
    mlx_enable_compile,
};

use crate::{
    closure::{MLXClosure, MLXFunc},
    VectorMLXArray,
};

struct CompileFunc<F, IN, OUT> {
    f: MLXClosure<IN, OUT>,
    inputs: VectorMLXArray,
    outputs: VectorMLXArray,
    id: usize,
    shapeless: bool,
    _data: PhantomData<F>,
}

impl<F, IN, OUT> CompileFunc<F, IN, OUT>
where
    F: MLXFunc<IN, OUT>,
    IN: for<'a> From<&'a VectorMLXArray> + Into<VectorMLXArray> + 'static,
    OUT: for<'b> From<&'b VectorMLXArray> + Into<VectorMLXArray> + 'static,
{
    pub fn new(f: F, shapeless: bool) -> Self {
        let id = f.id();
        Self {
            f: MLXClosure::new(f),
            inputs: VectorMLXArray::new(),
            outputs: VectorMLXArray::new(),
            id: id,
            shapeless: shapeless,
            _data: PhantomData::default(),
        }
    }

    pub fn apply(&self, input: IN) -> OUT {
        let constans: Vec<u64> = vec![];
        let handle = unsafe {
            mlx_detail_compile(
                self.f.as_ptr(),
                self.id,
                self.shapeless,
                constans.as_ptr(),
                constans.len(),
            )
        };
        MLXClosure::from_raw(handle).apply(input)
    }

    // pub fn apply(in: IN) -> OUT {

    // }
}

impl<F, IN, OUT> Drop for CompileFunc<F, IN, OUT> {
    fn drop(&mut self) {
        unsafe {
            mlx_detail_compile_erase(self.id);
        }
    }
}
pub fn compile<F, IN, OUT>(f: F, shapeless: bool) -> impl Fn(IN) -> OUT
where
    F: MLXFunc<IN, OUT>,
    IN: for<'a> From<&'a VectorMLXArray> + Into<VectorMLXArray> + 'static,
    OUT: for<'b> From<&'b VectorMLXArray> + Into<VectorMLXArray> + 'static,
{
    let cf = CompileFunc::new(f, shapeless);
    move |input| cf.apply(input)
}

pub fn enable_compile() {
    unsafe { mlx_enable_compile() }
}

pub fn disable_compile() {
    unsafe { mlx_disable_compile() }
}
