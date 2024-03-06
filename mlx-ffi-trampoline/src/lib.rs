use proc_macro::TokenStream;
use quote::format_ident;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn unary_ffi_trampoline(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    // Extract information from the input function
    let fn_name = &input_fn.sig.ident;
    let trampoline_name = format_ident!("{}_trampoline", fn_name);
    // let fn_block = &input_fn.block;
    // let fn_inputs = &input_fn.sig.inputs;
    // let fn_output = &input_fn.sig.output;

    // Generate the trampoline function
    let trampoline_fn = quote! {

        // Include the original function
        #input_fn

        #[no_mangle]
        pub extern "C" fn #trampoline_name(x: mlx_sys::mlx_array) -> mlx_sys::mlx_array {
            let input = mlx_rust::MLXArray::from_raw(x);
            let result = #fn_name(&input);
            let r = result.as_ptr();
            core::mem::forget(result);
            core::mem::forget(input);
            r
        }
    };

    // Return the generated trampoline function as a TokenStream
    TokenStream::from(trampoline_fn)
}

#[proc_macro_attribute]
pub fn vector_ffi_trampoline(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    // Extract information from the input function
    let fn_name = &input_fn.sig.ident;
    let trampoline_name = format_ident!("{}_trampoline", fn_name);
    // let fn_block = &input_fn.block;
    // let fn_inputs = &input_fn.sig.inputs;
    // let fn_output = &input_fn.sig.output;

    // Generate the trampoline function
    let trampoline_fn = quote! {

        // Include the original function
        #input_fn

        #[no_mangle]
        pub extern "C" fn #trampoline_name(x: mlx_sys::mlx_vector_array) -> mlx_sys::mlx_vector_array {
            let input = mlx_rust::VectorMLXArray::from_raw(x);
            let result = #fn_name(&input);
            let r = result.as_ptr();
            core::mem::forget(result);
            core::mem::forget(input);
            r
        }
    };

    // Return the generated trampoline function as a TokenStream
    TokenStream::from(trampoline_fn)
}
