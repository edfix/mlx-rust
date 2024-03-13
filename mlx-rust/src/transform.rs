use std::marker::PhantomData;

use mlx_sys::{mlx_closure_value_and_grad, mlx_closure_value_and_grad_, mlx_closure_value_and_grad_apply, mlx_free, mlx_jvp, mlx_value_and_grad, mlx_vector_vector_array_get, mlx_vjp};

use crate::closure::{MLXClosure, MLXFunc};
use crate::object::MLXObject;
use crate::VectorMLXArray;

/// Compute the Jacobian-vector product.
///
/// Computes the product of the `cotangents` with the Jacobian of a
/// function `f` evaluated at `primals`.
/// return (out, gradient)
pub fn jvp<IN, OUT>(
    f: impl MLXFunc<IN, OUT>,
    primals: IN,
    tangents: IN
) -> (OUT, VectorMLXArray) where
    IN: for<'a> From<&'a VectorMLXArray> + Into<VectorMLXArray>,
    OUT: for<'b> From<&'b VectorMLXArray> + Into<VectorMLXArray> {

    let closure = MLXClosure::new(f);
     unsafe {
         let vector_pair = mlx_jvp(
             closure.as_ptr(),
             primals.into().as_ptr(),
             tangents.into().as_ptr()
         );
         let out = mlx_vector_vector_array_get(vector_pair, 0);
         let gradient = mlx_vector_vector_array_get(vector_pair, 1);
         mlx_free(vector_pair as *mut ::std::os::raw::c_void);
         ((&VectorMLXArray::from_raw(out)).into(), VectorMLXArray::from_raw(gradient))
    }
}

/// Compute the vector-Jacobian product.
///
/// Computes the product of the `cotangents` with the Jacobian of a
/// function `f` evaluated at `primals`.
/// return (out, gradient)
pub fn vjp<IN, OUT>(
    f: impl MLXFunc<IN, OUT>,
    primals: IN,
    cotangents: IN
) -> (OUT, VectorMLXArray) where
    IN: for<'a> From<&'a VectorMLXArray> + Into<VectorMLXArray>,
    OUT: for<'b> From<&'b VectorMLXArray> + Into<VectorMLXArray> {

    let closure = MLXClosure::new(f);
    unsafe {
        let vector_pair = mlx_vjp(
            closure.as_ptr(),
            primals.into().as_ptr(),
            cotangents.into().as_ptr()
        );
        let out = mlx_vector_vector_array_get(vector_pair, 0);
        let gradient = mlx_vector_vector_array_get(vector_pair, 1);
        mlx_free(vector_pair as *mut ::std::os::raw::c_void);
        ((&VectorMLXArray::from_raw(out)).into(), VectorMLXArray::from_raw(gradient))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValueAndGrad<IN, OUT> {
    inner: MLXObject<mlx_closure_value_and_grad_>,
    _data: PhantomData<(IN, OUT)>
}

impl <IN, OUT> ValueAndGrad<IN, OUT> {
    fn from_raw(handle: mlx_closure_value_and_grad) -> ValueAndGrad<IN, OUT> {
        Self {
            inner: MLXObject::from_raw(handle),
            _data: PhantomData::default()
        }
    }
}

impl<IN, OUT>  ValueAndGrad<IN, OUT>
    where IN: Into<VectorMLXArray>,
          OUT: for<'a> From<&'a VectorMLXArray>, Self: 'static {
    pub fn apply(&self, input: IN) -> (OUT, VectorMLXArray) {
        let i: VectorMLXArray = input.into();
        unsafe {
            let vector_pair = mlx_closure_value_and_grad_apply(self.inner.as_ptr(), i.as_ptr());
            let out = mlx_vector_vector_array_get(vector_pair, 0);
            let gradient = mlx_vector_vector_array_get(vector_pair, 1);
            mlx_free(vector_pair as *mut ::std::os::raw::c_void);
            ((&VectorMLXArray::from_raw(out)).into(), VectorMLXArray::from_raw(gradient))
        }
    }
}

pub fn value_and_grad<IN, OUT, F>(f: F) -> ValueAndGrad<IN, OUT>
    where
        F: MLXFunc<IN, OUT>,
        IN: for<'a> From<&'a VectorMLXArray> + Into<VectorMLXArray>,
        OUT: for<'b> From<&'b VectorMLXArray> + Into<VectorMLXArray>
{
    let closure = MLXClosure::new(f);
        let args = vec![0];
        let handle = unsafe {
            mlx_value_and_grad(closure.as_ptr(), args.as_ptr(), args.len())
        };
        ValueAndGrad::from_raw(handle)
}

#[derive(Clone, Debug, PartialEq)]
pub struct GradFunc<IN, OUT>(ValueAndGrad<IN, OUT>);

impl<IN, OUT> GradFunc<IN, OUT> {
    fn new(value_and_grad: ValueAndGrad<IN, OUT>) -> Self {
        Self(value_and_grad)
    }
}

impl<IN, OUT> GradFunc<IN, OUT> where
    IN: for<'a> From<&'a VectorMLXArray> + Into<VectorMLXArray> + 'static,
    OUT: for<'b> From<&'b VectorMLXArray> + Into<VectorMLXArray> + 'static
{
    pub fn apply(&self, input: IN) -> VectorMLXArray {
        let (_, gradient) = self.0.apply(input);
        gradient
    }
}

impl <IN, OUT> MLXFunc<IN, VectorMLXArray> for  GradFunc<IN, OUT> where
    IN: for<'a> From<&'a VectorMLXArray> + Into<VectorMLXArray> + 'static,
    OUT: for<'b> From<&'b VectorMLXArray> + Into<VectorMLXArray> + 'static {
    fn apply(&self, input: IN) -> VectorMLXArray {
        self.apply(input)
    }
}


pub fn grad<IN, OUT, F>(f: F) -> GradFunc<IN, OUT> where
    F: MLXFunc<IN, OUT>,
    IN: for<'a> From<&'a VectorMLXArray> + Into<VectorMLXArray> + 'static,
    OUT: for<'b> From<&'b VectorMLXArray> + Into<VectorMLXArray> + 'static {
    let func = value_and_grad(f);
    GradFunc::new(func)
}



#[cfg(test)]
mod tests {
    use crate::MLXArray;
    use crate::transform::{grad, jvp, value_and_grad, vjp};

//     mlx_array inc_fun(mlx_array in) {
// mlx_array y = mlx_array_from_float(1.0);
// mlx_array res = mlx_add(in, y, MLX_GPU_STREAM);
// mlx_free(y);
// return res;
// }

    fn inc_fun(input: MLXArray) -> MLXArray {
        input + 1.0 + 2.0
    }

    #[test]
    fn test_jvp() {
        for _ in 0..1000 {
            let (out, gradient) = jvp(
                inc_fun,
                1.0.into(),
                1.0.into()
            );
            // println!("out: {}", out.get(0).unwrap());
            assert_eq!(4.0, out.to_scalar::<f32>().unwrap());
            assert_eq!(1.0, gradient.get(0).unwrap().to_scalar::<f32>().unwrap())
        }
    }

    #[test]
    fn test_vjp() {
        for _ in 0..1000 {
            let (out, gradient) = vjp(
                inc_fun,
                1.0.into(),
                1.0.into()
            );
            // println!("out: {}", out.get(0).unwrap());
            assert_eq!(4.0, out.to_scalar::<f32>().unwrap());
            assert_eq!(1.0, gradient.get(0).unwrap().to_scalar::<f32>().unwrap())
        }
    }

    #[test]
    fn test_value_grad() {
        for _ in 0..1000 {
            let (out, gradient) = value_and_grad(inc_fun).apply(1.0.into());
            // println!("out: {}", out.get(0).unwrap());
            assert_eq!(4.0, out.to_scalar::<f32>().unwrap());
            assert_eq!(1.0, gradient.get(0).unwrap().to_scalar::<f32>().unwrap())
        }
    }

    #[test]
    fn test_grad() {
        for _ in 0..1000 {
            let gradient = grad(grad(inc_fun)).apply(1.0.into());
            assert_eq!(0.0, gradient.get(0).unwrap().to_scalar::<f32>().unwrap());

            let gradient0 = grad(inc_fun).apply(1.0.into());
            assert_eq!(1.0, gradient0.get(0).unwrap().to_scalar::<f32>().unwrap())
        }
    }
}