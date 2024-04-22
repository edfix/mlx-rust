#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub use array::*;
pub use vector_array::*;

pub mod array;
pub mod array_op;
pub mod closure;
pub mod compile;
pub mod device;
pub mod from_array;
pub mod io;
mod object;
pub mod random;
pub mod stream;
mod string;
pub mod to_array;
pub mod transform;
pub mod r#type;
pub mod vector_array;
pub mod fast;
pub mod module;

