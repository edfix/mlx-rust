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

## Building MLX C

To build the MLX C library, follow these steps:

```sh
git clone https://github.com/edfix/mlx-rust.git
cd mlx-c
cargo build
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
