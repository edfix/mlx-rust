use mlx_ffi_trampoline::{unary_ffi_trampoline, vector_ffi_trampoline};
use mlx_rust::{closure::MLXClosure, MLXArray, VectorMLXArray};

#[unary_ffi_trampoline]
pub fn f1(input: &MLXArray) -> MLXArray {
    input + 2
}

#[vector_ffi_trampoline]
pub fn vector_f1(input: &VectorMLXArray) -> VectorMLXArray {
    input.clone()
}

pub fn main() {
    // apply unary call
    let f = MLXClosure::new_unary(f1_trampoline);
    let result = f.apply(2);
    let r = result.get(0).unwrap().to_scalar::<i32>().unwrap();
    println!("result: {}", r);

    //apply vector call
    let f1 = MLXClosure::new(vector_f1_trampoline);
    let result = f1.apply(2);
    let r1 = result.get(0).unwrap().to_scalar::<i32>().unwrap();
    println!("result: {}", r1)
}
