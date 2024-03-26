extern crate bindgen;

use std::env;
use std::path::PathBuf;
// use std::process::Command;

fn main() {
    // // This is the directory where the `c` library is located.
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_path = PathBuf::from(manifest_dir);

    // // Initialize and update the submodule
    // let status = Command::new("git")
    //     .args(&["submodule", "update", "--init", "--recursive"])
    //     .current_dir(&project_path) // Run the command in the project directory
    //     .status()
    //     .expect("Failed to run git submodule update");

    // if !status.success() {
    //     panic!("Failed to initialize or update git submodules");
    // }

    let libdir_path = project_path
        .join("mlx-c")
        .canonicalize()
        .expect("cannot canonicalize path");

    let dst = cmake::build(&libdir_path);

    // This is the path to the `c` headers file.
    let headers_path_str = libdir_path.join("mlx/c/mlx.h");
    let transform_header_str = libdir_path.join("mlx/c/transforms_impl.h");

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={}", dst.join("lib").display());

    // Tell cargo to tell rustc to link our `mlxc` library. Cargo will
    // automatically know it must look for a `libmlxc.a` file.
    println!("cargo:rustc-link-lib=mlxc");
    println!("cargo:rustc-link-lib=mlx");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=dylib=c++");

    let bindings = bindgen::Builder::default()
        .header(headers_path_str.to_str().unwrap())
        .header(transform_header_str.to_str().unwrap())
        .clang_arg(format!("-I{}", libdir_path.to_str().unwrap()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings!");
}
