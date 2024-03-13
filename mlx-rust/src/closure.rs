use std::marker::PhantomData;
use std::mem::{forget, transmute};

use mlx_sys::{mlx_closure, mlx_closure_, mlx_closure_apply, mlx_closure_new_with_payload, mlx_vector_array};

use crate::{MLXArray, object::MLXObject, VectorMLXArray};

#[derive(PartialEq, Debug)]
pub struct MLXClosure<IN, OUT> {
    handle: MLXObject<mlx_closure_>,
    _data: PhantomData<(IN, OUT)>
}

pub(crate) struct FFiCallback(Box<dyn Fn(&VectorMLXArray) -> VectorMLXArray>);

impl FFiCallback {
    pub fn new(f: impl Fn(&VectorMLXArray) -> VectorMLXArray + 'static) -> Self {
        Self(Box::new(f))
    }
}

impl <T: Fn(&MLXArray) -> MLXArray + 'static> From<T> for FFiCallback {
    fn from(value: T) -> Self {
        let wrapper = move |input: &VectorMLXArray| {
            let r = value(&input.get(0).unwrap());
            VectorMLXArray::from_array(r)
        };
        FFiCallback::new(wrapper)
    }
}

pub trait MLXFunc<IN, OUT> where OUT: Into<VectorMLXArray>, IN: for<'a> From<&'a VectorMLXArray>, Self: 'static {
    fn apply(&self, input: IN) -> OUT;
}

fn to_callback<IN, OUT>(f: impl MLXFunc<IN, OUT> + Sized) -> FFiCallback
    where OUT: Into<VectorMLXArray>,
          IN: for<'a> From<&'a VectorMLXArray> {
    let wrapper = move |input: &VectorMLXArray| {
        let r = f.apply(input.into());
        let result: VectorMLXArray = r.into();
        result
    };
    FFiCallback::new(wrapper)
}



impl <'a> From<&'a VectorMLXArray> for MLXArray {
    fn from(value: &'a VectorMLXArray) -> Self {
        value.get(0).unwrap()
    }
}

impl <T> MLXFunc<MLXArray, MLXArray> for T where T: Fn(MLXArray) -> MLXArray + 'static {
    fn apply(&self, input: MLXArray) -> MLXArray {
        self(input)
    }
}

// impl <IN, T> MLXFunc<IN, VectorMLXArray> for T where T: Fn(IN) -> VectorMLXArray + 'static  {
//     fn apply(&self, input: IN) -> VectorMLXArray {
//         self(&input)
//     }
// }

impl <T> MLXFunc<VectorMLXArray, VectorMLXArray> for T where T: Fn(&VectorMLXArray) -> VectorMLXArray + 'static  {
    fn apply(&self, input: VectorMLXArray) -> VectorMLXArray {
        self(&input)
    }
}


impl <IN, OUT> MLXClosure<IN, OUT>
    where IN: for<'a> From<&'a VectorMLXArray> + Into<VectorMLXArray>,
    OUT: for<'b> From<&'b VectorMLXArray> + Into<VectorMLXArray>
{
    pub fn new(f: impl MLXFunc<IN, OUT>) -> MLXClosure<IN, OUT> {
        Self::from_FFiCallback(to_callback(f))
    }

    fn from_FFiCallback(f: FFiCallback) -> Self {
        let fc: Box<FFiCallback> = Box::new(f);
        let payload  = Box::into_raw(fc) as *mut ::std::os::raw::c_void;
        extern "C" fn trampoline(input: mlx_vector_array,
                                 payload: *mut ::std::os::raw::c_void,) -> mlx_vector_array {
            let callback: Box<FFiCallback> = unsafe {transmute(payload)};
            let i = VectorMLXArray::from_raw(input);
            let r = callback.0(&i);
            let result = r.as_ptr();
            forget(callback);
            forget(r);
            forget(i);
            result
        }

        unsafe extern "C" fn free(arg1: *mut ::std::os::raw::c_void) {
            let _: Box<FFiCallback> = Box::from_raw(arg1 as *mut _);
        }
        let handle = unsafe {mlx_closure_new_with_payload(Some(trampoline), payload, Some(free))};
        Self::from_raw(handle)
    }

    pub fn apply(&self, input: IN) -> OUT {
        let args: VectorMLXArray = input.into();
        let handle = unsafe { mlx_closure_apply(self.as_ptr(), args.as_ptr()) };
        let vector_array = VectorMLXArray::from_raw(handle);
        (&vector_array).into()
    }


    pub(crate) fn from_raw(handle: mlx_closure) -> MLXClosure<IN, OUT> {
        Self {
            handle: MLXObject::from_raw(handle),
            _data: PhantomData::default()
        }
    }

    pub(crate) fn as_ptr(&self) -> mlx_closure {
        self.handle.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use mlx_sys::mlx_vjp;

    use crate::{MLXArray, VectorMLXArray};

    use super::MLXClosure;

    fn unary_call(x: MLXArray) -> MLXArray {
        x.clone() + 3
    }

    // extern "C" fn unary_call_wrap(x: mlx_array) -> mlx_array {
    //     let input = MLXArray::from_raw(x);
    //     let result = unary_call(&input);
    //     let r = result.as_ptr();
    //     forget(result);
    //     forget(input);
    //     r
    // }

    fn vector_call(x: &VectorMLXArray) -> VectorMLXArray {
        x.clone()
    }

    // extern "C" fn vector_call_wrap(x: mlx_vector_array) -> mlx_vector_array {
    //     let input = VectorMLXArray::from_raw(x);
    //     let result = vector_call(&input);
    //     let r = result.as_ptr();
    //     forget(result);
    //     forget(input);
    //     r
    // }

    // #[test]
    // fn test_apply_vector_mlx_closure() {
    //     for _ in 0..1000 {
    //         let f = MLXClosure::new(vector_call_wrap);
    //         let result = f.apply(2);
    //         let r = result.get(0).unwrap().to_scalar::<i32>().unwrap();
    //         assert_eq!(2, r);
    //         assert_eq!(1, result.len())
    //     }
    // }

    #[test]
    fn test_apply_vector_() {
        for _ in 0..1000 {
            let f = MLXClosure::new(vector_call);
            let result = f.apply(2.into());
            let r = result.get(0).unwrap().to_scalar::<i32>().unwrap();
            assert_eq!(2, r);
            assert_eq!(1, result.len())
        }
    }

    #[test]
    fn test_apply_unary_mlx_closure() {
        for _ in 0..1000 {
            let f = MLXClosure::new(unary_call);
            let result = f.apply(2.into());
            let r = result.to_scalar::<i32>().unwrap();
            assert_eq!(5, r)
        }
    }

    #[test]
    fn test_compile() {
        let f = MLXClosure::new(vector_call);
        let x: VectorMLXArray = 2.into();
        let v: VectorMLXArray = 3.into();
        unsafe { mlx_vjp(f.as_ptr(), x.as_ptr(), v.as_ptr()) };

        println!("compiled ....")
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
