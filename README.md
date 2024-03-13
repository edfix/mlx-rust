# Rust Bindings for MLX

Welcome to the Rust Bindings for MLX project! This repository contains Rust language bindings for MLX, an array framework designed for machine learning research on Apple silicon.

## Overview

mlx-rust use MLX C API for binding.These Rust bindings provide a safe and efficient interface to MLX, allowing researchers and developers to harness the power of MLX within Rust applications.

## Prerequisites

Before using the Rust Bindings for MLX, ensure you have the following installed on your Apple silicon Mac:

- Rust (latest stable version)
- Cargo (Rust's package manager)
- Git (for cloning the repository)
- CMake (for compiling MLX C API)

## Building mlx-rust

To build the mlx-rust library, follow these steps:

```sh
git clone https://github.com/edfix/mlx-rust.git
cd mlx-rust
git submodule add https://github.com/ml-explore/mlx-c.git ./mlx-sys/mlx-c
cargo build
```

Installation

```sh
cargo add mlx-rust
cargo add mlx-sys
```

## Code snippets

```rust
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


```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
