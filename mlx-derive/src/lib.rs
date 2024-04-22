extern crate proc_macro;

use proc_macro::TokenStream;

use proc_macro2::Span;
use quote::ToTokens;
use syn::{DeriveInput, Path, Type};

#[derive(Debug, deluxe::ExtractAttributes)]
#[deluxe(attributes(module))]
#[deluxe(default)]
struct ContainerOpts {
    #[deluxe(rename = input)]
    input_ty: Option<Type>,

    #[deluxe(rename = output)]
    output_ty: Option<Type>,

    #[deluxe(rename = crate)]
    crate_root: Path,

    trainable: bool,
}
impl Default for ContainerOpts {
    fn default() -> Self {
        Self {
            input_ty: None,
            output_ty: None,
            crate_root: syn::parse_quote!(mlx_rust),
            trainable: true,
        }
    }
}

#[derive(Debug, deluxe::ParseAttributes)]
#[deluxe(attributes(param))]
struct FieldOpts<'t> {
    #[deluxe(container)]
    field: &'t syn::Field,
    #[deluxe(default)]
    rename: Option<String>,
    #[deluxe(default)]
    skip: bool,
}

#[proc_macro_derive(Module, attributes(module, param))]
pub fn module(item: TokenStream) -> TokenStream {
    let mut input: DeriveInput = syn::parse(item).expect("syn::parse ok");

    let errors = deluxe::Errors::new();
    let ContainerOpts {
        input_ty,
        output_ty,
        crate_root,
        trainable,
    } = deluxe::extract_attributes_optional(&mut input, &errors);

    let mut field_opts: Vec<FieldOpts> = Vec::new();
    let mut is_unit_struct = false;
    if let syn::Data::Struct(s) = &mut input.data {
        match &mut s.fields {
            syn::Fields::Named(fields) => {
                for field in fields.named.iter_mut() {
                    match deluxe::parse_attributes(field) {
                        Ok(f_opts) => field_opts.push(f_opts),
                        Err(e) => errors.push_syn(e),
                    }
                }
            }
            syn::Fields::Unit => is_unit_struct = true,
            syn::Fields::Unnamed(_) => errors.push(Span::call_site(), "tuple is not supported"),
        }
    }
    if !errors.is_empty() {
        return errors.into_token_stream().into();
    }

    let receiver_name = &input.ident;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();

    let call_fwd = quote::quote! {
                self.fwd(input)
            };
    let input_ty = input_ty.unwrap_or_else(|| {
        syn::parse_quote! {
            ::#crate_root::MLXArray
        }
    });

    let output_ty = output_ty.unwrap_or_else(|| {
        syn::parse_quote! {
            ::#crate_root::MLXArray
        }
    });

    // let call_fwd = match &input_ty {
    //     Type::Path(_) | Type::Array(_) => {
    //         quote::quote! {
    //             self.fwd(input)
    //         }
    //     }
    //     Type::Tuple(tuple) => {
    //         let args: Vec<_> = tuple
    //             .elems
    //             .iter()
    //             .enumerate()
    //             .map(|(i, t)| {
    //                 let arg = Ident::new(&format!("a{i}"), t.span());
    //                 quote::quote! {
    //                     #arg
    //                 }
    //             })
    //             .collect();
    //
    //         quote::quote! {
    //             let (#(#args,)*) = input;
    //             self.fwd(#(::#crate_root::nn::ToApplyArg::to_arg(#args),)*)
    //         }
    //     }
    //     _ => panic!("unsupported module input type"),
    // };

    let module_impls = if is_unit_struct || !trainable {
        quote::quote! {
            impl #impl_generics ::#crate_root::module::Module for #receiver_name #type_generics #where_clause {
                type Input = #input_ty;
                // type Output = #output_ty;

                #[inline]
                fn forward(&self, input: #input_ty) -> #output_ty {
                    #call_fwd
                }
                // fn gather_params(&self, params: &mut std::collections::HashMap<usize, ::#crate_root::Tensor>) {}
                // fn update_params(&self, params: &mut std::collections::HashMap<usize, ::#crate_root::Tensor>) {}
                // fn gather_named_params(&self, prefix: &str, params: &mut std::collections::HashMap<String, ::#crate_root::Tensor>) {}
                fn update_named_params(&mut self, prefix: &str, params: &mut std::collections::HashMap<String, ::#crate_root::MLXArray>) {}
            }
            //
            // impl #impl_generics ::#crate_root::ValueSpec for #receiver_name #type_generics #where_clause {
            //     type Kind = ::#crate_root::ModuleValue;
            //     type Tensors = ();
            //     type Gradient = ();
            // }
            //
            // impl #impl_generics ::#crate_root::nn::NonTrainableModule for #receiver_name #type_generics #where_clause {}
        }
    } else {
        // let update_params: Vec<_> = field_opts
        //     .iter()
        //     .filter(|f| !f.skip)
        //     .map(|f| {
        //         let field_name = f.field.ident.as_ref().unwrap();
        //         quote::quote! {
        //             ::#crate_root::nn::WithParams::update_by_id(&self.#field_name, params);
        //         }
        //     })
        //     .collect();

        // let gather_params: Vec<_> = field_opts
        //     .iter()
        //     .filter(|f| !f.skip)
        //     .map(|f| {
        //         let field_name = f.field.ident.as_ref().unwrap();
        //         quote::quote! {
        //             ::#crate_root::nn::WithParams::gather_by_id(&self.#field_name, params);
        //         }
        //     })
        //     .collect();

        let update_named_params: Vec<_> = field_opts
            .iter()
            .filter(|f| !f.skip)
            .map(|f| {
                let field_name = f.field.ident.as_ref().unwrap();
                let f_name = field_name.to_string();
                let param_name = f.rename.as_ref().unwrap_or(&f_name);
                quote::quote! {
                    ::#crate_root::module::WithParams::update_by_name(&mut self.#field_name, params, prefix, #param_name);
                }
            })
            .collect();

        // let gather_named_params: Vec<_> = field_opts
        //     .iter()
        //     .filter(|f| !f.skip)
        //     .map(|f| {
        //         let field_name = f.field.ident.as_ref().unwrap();
        //         let f_name = field_name.to_string();
        //         let param_name = f.rename.as_ref().unwrap_or(&f_name);
        //         quote::quote! {
        //             ::#crate_root::nn::WithParams::gather_by_name(&self.#field_name, params, prefix, #param_name);
        //         }
        //     })
        //     .collect();

        quote::quote! {
            impl #impl_generics ::#crate_root::module::Module for #receiver_name #type_generics #where_clause {
                type Input = #input_ty;
                // type Output = #output_ty;

                #[inline]
                fn forward(&self, input: #input_ty) -> #output_ty {
                    #call_fwd
                }

                // fn gather_params(&self, params: &mut std::collections::HashMap<usize, ::#crate_root::Tensor>) {
                //     #(#gather_params)*
                // }
                //
                // fn update_params(&self, params: &mut std::collections::HashMap<usize, ::#crate_root::Tensor>) {
                //     #(#update_params)*
                // }
                //
                // fn gather_named_params(&self, prefix: &str, params: &mut std::collections::HashMap<String, ::#crate_root::Tensor>) {
                //     #(#gather_named_params)*
                // }

                fn update_named_params(&mut self, prefix: &str, params: &mut std::collections::HashMap<String, ::#crate_root::MLXArray>) {
                    #(#update_named_params)*
                }
            }

            // impl #impl_generics ::#crate_root::ValueSpec for #receiver_name #type_generics #where_clause {
            //     type Kind = ::#crate_root::ModuleValue;
            //     type Tensors = std::collections::HashMap<usize, Tensor>;
            //     type Gradient = std::collections::HashMap<usize, Tensor>;
            // }
            //
            // impl #impl_generics ::#crate_root::nn::TrainableModule for #receiver_name #type_generics #where_clause {}
        }
    };

    module_impls.into()
}


// use proc_macro::TokenStream;
// use quote::format_ident;
// use quote::quote;
// use syn::{parse_macro_input, ItemFn};
//
// #[proc_macro_attribute]
// pub fn unary_ffi_trampoline(_attr: TokenStream, item: TokenStream) -> TokenStream {
//     let input_fn = parse_macro_input!(item as ItemFn);
//
//     // Extract information from the input function
//     let fn_name = &input_fn.sig.ident;
//     let trampoline_name = format_ident!("{}_trampoline", fn_name);
//     // let fn_block = &input_fn.block;
//     // let fn_inputs = &input_fn.sig.inputs;
//     // let fn_output = &input_fn.sig.output;
//
//     // Generate the trampoline function
//     let trampoline_fn = quote! {
//
//         // Include the original function
//         #input_fn
//
//         #[no_mangle]
//         pub extern "C" fn #trampoline_name(x: mlx_sys::mlx_array) -> mlx_sys::mlx_array {
//             let input = mlx_rust::MLXArray::from_raw(x);
//             let result = #fn_name(&input);
//             let r = result.as_ptr();
//             core::mem::forget(result);
//             core::mem::forget(input);
//             r
//         }
//     };
//
//     // Return the generated trampoline function as a TokenStream
//     TokenStream::from(trampoline_fn)
// }
//
// #[proc_macro_attribute]
// pub fn vector_ffi_trampoline(_attr: TokenStream, item: TokenStream) -> TokenStream {
//     let input_fn = parse_macro_input!(item as ItemFn);
//
//     // Extract information from the input function
//     let fn_name = &input_fn.sig.ident;
//     let trampoline_name = format_ident!("{}_trampoline", fn_name);
//     // let fn_block = &input_fn.block;
//     // let fn_inputs = &input_fn.sig.inputs;
//     // let fn_output = &input_fn.sig.output;
//
//     // Generate the trampoline function
//     let trampoline_fn = quote! {
//
//         // Include the original function
//         #input_fn
//
//         #[no_mangle]
//         pub extern "C" fn #trampoline_name(x: mlx_sys::mlx_vector_array) -> mlx_sys::mlx_vector_array {
//             let input = mlx_rust::VectorMLXArray::from_raw(x);
//             let result = #fn_name(&input);
//             let r = result.as_ptr();
//             core::mem::forget(result);
//             core::mem::forget(input);
//             r
//         }
//     };
//
//     // Return the generated trampoline function as a TokenStream
//     TokenStream::from(trampoline_fn)
// }
