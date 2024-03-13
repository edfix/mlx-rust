use mlx_rust::{closure::MLXClosure, MLXArray, VectorMLXArray};
use mlx_rust::transform::{grad, value_and_grad};

pub fn f1(input: MLXArray) -> MLXArray {
    input + 2
}

pub fn vector_f1(input: &VectorMLXArray) -> VectorMLXArray {
    input.clone()
}

pub fn main() {
    // apply unary call
    let f = MLXClosure::new(f1);
    let result = f.apply(2.into());
    let r = result.to_scalar::<i32>().unwrap();
    println!("unary result: {}", r);

    //apply vector call
    let f2 = MLXClosure::new(vector_f1);
    let result = f2.apply(2.into());
    let r1 = result.get(0).unwrap().to_scalar::<i32>().unwrap();
    println!("result: {}", r1);

    //grad
    let r = grad(f1).apply(1.into());
    println!("grad: {}", r.get(0).unwrap());

    let r = grad(grad(f1)).apply(1.into());
    println!("hessian : {}", r.get(0).unwrap());

    //value and grad
    let vg =  value_and_grad(f1);
    let (out, gradient) = vg.apply(1.into());
    println!("value and grad: ({}, {})", out, gradient.get(0).unwrap())

}
