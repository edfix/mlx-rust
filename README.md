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
cargo add mlx-ffi-trampoline
cargo add mlx-sys
```

## Code snippets

```rust
use ffi_trampoline::{unary_ffi_trampoline, vector_ffi_trampoline};
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
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
