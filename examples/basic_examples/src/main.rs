use mlx_rust::array_op::{erf, sin, sqrt, square};
use mlx_rust::closure::MLXFunc;
use mlx_rust::compile::{compile, disable_compile, enable_compile};
use mlx_rust::stream::get_default_stream;
use mlx_rust::transform::{grad, value_and_grad};
use mlx_rust::{closure::MLXClosure, MLXArray, VectorMLXArray};
use std::f32::consts::PI;
use std::time::SystemTime;

pub fn f1(input: MLXArray) -> MLXArray {
    input + 2
}

pub fn gelu(x: MLXArray) -> MLXArray {
    x.clone() * (1 + erf(x / sqrt(2.into()))) / 2
}

pub fn vector_f1(input: &VectorMLXArray) -> VectorMLXArray {
    input.clone()
}

fn loss_fn(w: MLXArray, x: MLXArray, y: MLXArray) -> MLXArray {
    let r = square(w * x - y).mean_all(false, get_default_stream());
    r
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

    let dfdx = grad(sin);
    println!("sin grad: {}", dfdx.apply(PI.into()).get(0).unwrap());

    let r = grad(grad(sin)).apply((PI / 2.0).into());
    println!("hessian : {}", r.get(0).unwrap());

    //value and grad
    let vg = value_and_grad(f1, &vec![0]);
    let (out, gradient) = vg.apply(1.into());
    println!("value and grad: ({}, {})", out, gradient.get(0).unwrap());

    let r = loss_fn.apply((
        1.0.into(),
        MLXArray::array(&[0.5, -0.5], &[2]),
        MLXArray::array(&[1.5, -1.5], &[2]),
    ));
    println!("r.....{}", r);

    let dfdw = grad(loss_fn);
    println!("start call dfdw");
    let dloss_dw = dfdw.apply((
        1.0.into(),
        MLXArray::array(&[0.5, -0.5], &[2]),
        MLXArray::array(&[1.5, -1.5], &[2]),
    ));
    println!("loss: {}", dloss_dw.get(0).unwrap());

    enable_compile();
    let one = MLXArray::ones::<f32>(&[32, 1000, 4096], get_default_stream());
    println!("start compile fn");
    measure(gelu, one.clone());
    measure(compile(gelu, false), one);
}

fn measure(f: impl Fn(MLXArray) -> MLXArray, x: MLXArray) {
    //warm up
    for _ in 0..10 {
        f(x.clone()).eval()
    }

    let iterations = 100;
    let now = SystemTime::now();
    for _ in 0..iterations {
        f(x.clone()).eval()
    }

    println!(
        "Time per iteration {:?}",
        now.elapsed().unwrap().checked_div(iterations)
    )
}
