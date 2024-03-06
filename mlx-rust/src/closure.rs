use mlx_sys::{
    mlx_array_, mlx_closure, mlx_closure_, mlx_closure_apply, mlx_closure_new,
    mlx_closure_new_unary, mlx_vector_array,
};

use crate::{object::MLXObject, VectorMLXArray};

#[derive(PartialEq, Debug)]
pub struct MLXClosure {
    handle: MLXObject<mlx_closure_>,
}

pub type UnaryFFICall = unsafe extern "C" fn(*mut mlx_array_) -> *mut mlx_array_;
pub type VecctorFFICall = unsafe extern "C" fn(arrs: mlx_vector_array) -> mlx_vector_array;

impl MLXClosure {
    pub fn new(f: VecctorFFICall) -> Self {
        let handle = unsafe { mlx_closure_new(Some(f)) };
        Self::from_raw(handle)
    }

    pub fn new_unary(f: UnaryFFICall) -> Self {
        let handle = unsafe { mlx_closure_new_unary(Some(f)) };
        Self::from_raw(handle)
    }

    pub fn apply(&self, value: impl Into<VectorMLXArray>) -> VectorMLXArray {
        let args: VectorMLXArray = value.into();
        let handle = unsafe { mlx_closure_apply(self.as_ptr(), args.as_ptr()) };
        VectorMLXArray::from_raw(handle)
    }

    pub(crate) fn from_raw(handle: mlx_closure) -> Self {
        Self {
            handle: MLXObject::from_raw(handle),
        }
    }

    pub(crate) fn as_ptr(&self) -> mlx_closure {
        self.handle.as_ptr()
    }
}

#[cfg(test)]
mod tests {

    use std::mem::forget;

    use mlx_sys::{
        mlx_array, mlx_array_from_float, mlx_closure_new_unary, mlx_compile, mlx_enable_compile,
        mlx_vector_array, mlx_vector_array_from_array,
    };

    use super::MLXClosure;
    use crate::{MLXArray, VectorMLXArray};

    fn unary_call(x: &MLXArray) -> MLXArray {
        x.clone() + 3
    }

    extern "C" fn unary_call_wrap(x: mlx_array) -> mlx_array {
        let input = MLXArray::from_raw(x);
        let result = unary_call(&input);
        let r = result.as_ptr();
        forget(result);
        forget(input);
        r
    }

    fn vector_call(x: &VectorMLXArray) -> VectorMLXArray {
        x.clone()
    }

    extern "C" fn vector_call_wrap(x: mlx_vector_array) -> mlx_vector_array {
        let input = VectorMLXArray::from_raw(x);
        let result = vector_call(&input);
        let r = result.as_ptr();
        forget(result);
        forget(input);
        r
    }

    #[test]
    fn test_apply_vector_mlx_closure() {
        for _ in 0..1000 {
            let f = MLXClosure::new(vector_call_wrap);
            let result = f.apply(2);
            let r = result.get(0).unwrap().to_scalar::<i32>().unwrap();
            assert_eq!(2, r);
            assert_eq!(1, result.len())
        }
    }

    #[test]
    fn test_apply_unary_mlx_closure() {
        for _ in 0..1000 {
            let f = MLXClosure::new_unary(unary_call_wrap);
            // unary_call.apply(&2.into());
            let result = f.apply(2);
            let r = result.get(0).unwrap().to_scalar::<i32>().unwrap();
            assert_eq!(5, r)
        }
    }

    // #[test]
    // fn test_apply_raw_mlx_closure() {
    // for _ in 0..1000 {
    //     unsafe {
    //         let i0 = mlx_array_from_float(12.0);
    //         let input = mlx_vector_array_from_array(i0);
    //         let closure = mlx_closure_new_unary(Some(unary_call_wrap));

    // mlx_enable_compile();
    // let compiled = mlx_compile(closure, false);
    // let output = mlx_closure_apply(closure, input);
    // }
    // let f = MLXClosure::new(test_double);
    // let result = f.apply(vec![2.into()]);
    // println!("size==: {}", result.len());
    // forget(result)
    // let r = result.get(0).unwrap().to_scalar::<i32>().unwrap();
    // println!("result: {:?}", r);
    // forget(result)

    // assert_eq!(4, r)
    // }
    // }
}
