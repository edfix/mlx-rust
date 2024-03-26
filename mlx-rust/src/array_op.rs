use mlx_sys::{
    mlx_add, mlx_addmm, mlx_cos, mlx_cosh, mlx_divide, mlx_erf, mlx_erfinv, mlx_exp, mlx_floor,
    mlx_log, mlx_log10, mlx_matmul, mlx_mean, mlx_mean_all, mlx_multiply, mlx_sigmoid, mlx_sign,
    mlx_sin, mlx_sinh, mlx_sqrt, mlx_square, mlx_subtract, mlx_tan, mlx_tanh, mlx_transpose_all,
};
use std::ops::{Add, Div, Mul, Sub};

use crate::array::MLXArray;
use crate::stream::{get_default_stream, MLXStream};

impl MLXArray {
    pub fn add_with_stream(&self, rhs: &MLXArray, stream: MLXStream) -> Self {
        let handle = unsafe { mlx_add(self.as_ptr(), rhs.as_ptr(), stream.as_ptr()) };
        Self::from_raw(handle)
    }

    pub fn sub_with_stream(&self, rhs: &MLXArray, stream: MLXStream) -> Self {
        let handle = unsafe { mlx_subtract(self.as_ptr(), rhs.as_ptr(), stream.as_ptr()) };
        Self::from_raw(handle)
    }

    pub fn div_with_stream(&self, rhs: &MLXArray, stream: MLXStream) -> Self {
        let handle = unsafe { mlx_divide(self.as_ptr(), rhs.as_ptr(), stream.as_ptr()) };
        Self::from_raw(handle)
    }

    pub fn mul_with_stream(&self, rhs: &MLXArray, stream: MLXStream) -> Self {
        let handle = unsafe { mlx_multiply(self.as_ptr(), rhs.as_ptr(), stream.as_ptr()) };
        Self::from_raw(handle)
    }

    pub fn mean(&self, axes: &[i32], keepdims: bool, stream: MLXStream) -> MLXArray {
        let handle = unsafe {
            mlx_mean(
                self.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                keepdims,
                stream.as_ptr(),
            )
        };
        MLXArray::from_raw(handle)
    }

    pub fn mean_all(&self, keepdims: bool, stream: MLXStream) -> MLXArray {
        let handle = unsafe { mlx_mean_all(self.as_ptr(), keepdims, stream.as_ptr()) };
        MLXArray::from_raw(handle)
    }

    pub fn matmul(&self, b: MLXArray, stream: MLXStream) -> MLXArray {
        let handle = unsafe { mlx_matmul(self.as_ptr(), b.as_ptr(), stream.as_ptr()) };
        MLXArray::from_raw(handle)
    }

    pub fn t(&self) -> MLXArray {
        let handle = unsafe { mlx_transpose_all(self.as_ptr(), get_default_stream().as_ptr()) };
        MLXArray::from_raw(handle)
    }
}

pub fn addmm(
    x: MLXArray,
    bias: MLXArray,
    weight: MLXArray,
    alpha: f32,
    beta: f32,
    stream: MLXStream,
) -> MLXArray {
    let handle = unsafe {
        mlx_addmm(
            bias.as_ptr(),
            x.as_ptr(),
            weight.t().as_ptr(),
            alpha,
            beta,
            stream.as_ptr(),
        )
    };
    MLXArray::from_raw(handle)
}

macro_rules! impl_unary_op {
    ($func_name:ident, $mlx_func:ident) => {
        pub fn $func_name(v: MLXArray) -> MLXArray {
            let handle = unsafe { $mlx_func(v.as_ptr(), get_default_stream().as_ptr()) };
            MLXArray::from_raw(handle)
        }
    };
}

// Usage of the macro to implement `log` and `log10` functions
impl_unary_op!(sigmoid, mlx_sigmoid);
impl_unary_op!(sign, mlx_sign);
impl_unary_op!(sin, mlx_sin);
impl_unary_op!(sinh, mlx_sinh);
impl_unary_op!(cos, mlx_cos);
impl_unary_op!(cosh, mlx_cosh);
impl_unary_op!(tan, mlx_tan);
impl_unary_op!(tanh, mlx_tanh);
impl_unary_op!(square, mlx_square);
impl_unary_op!(sqrt, mlx_sqrt);
impl_unary_op!(log, mlx_log);
impl_unary_op!(log10, mlx_log10);
impl_unary_op!(erf, mlx_erf);
impl_unary_op!(erfinv, mlx_erfinv);
impl_unary_op!(exp, mlx_exp);
impl_unary_op!(floor, mlx_floor);
// impl_unary_op!(t, mlx_transpose_all);

macro_rules! impl_lhs_binary_trait {
    ($type:ty, $trait:ident, $op:ident) => {
        impl $trait<MLXArray> for $type {
            type Output = MLXArray;

            fn $op(self, rhs: MLXArray) -> Self::Output {
                let l: MLXArray = self.into();
                l.$op(&rhs)
            }
        }

        impl $trait<&MLXArray> for $type {
            type Output = MLXArray;

            fn $op(self, rhs: &MLXArray) -> Self::Output {
                let l: MLXArray = self.into();
                l.$op(rhs)
            }
        }
    };
}

macro_rules! impl_binary_trait {
    ($trait:ident, $op:ident, $method: ident) => {
        impl<R: Into<MLXArray>> $trait<R> for MLXArray {
            type Output = MLXArray;

            fn $op(self, rhs: R) -> Self::Output {
                let stream = get_default_stream();
                let r: MLXArray = rhs.into();
                self.$method(&r, stream)
            }
        }

        impl $trait<&MLXArray> for MLXArray {
            type Output = MLXArray;

            fn $op(self, rhs: &MLXArray) -> Self::Output {
                let stream = get_default_stream();
                self.$method(rhs, stream)
            }
        }

        impl<R: Into<MLXArray>> $trait<R> for &MLXArray {
            type Output = MLXArray;

            fn $op(self, rhs: R) -> Self::Output {
                let stream = get_default_stream();
                let r: MLXArray = rhs.into();
                self.$method(&r, stream)
            }
        }

        impl $trait<&MLXArray> for &MLXArray {
            type Output = MLXArray;

            fn $op(self, rhs: &MLXArray) -> Self::Output {
                let stream = get_default_stream();
                self.$method(rhs, stream)
            }
        }
        // impl_lhs_binary_trait!(u8, $trait, $op);
        // impl_lhs_binary_trait!(u16, $trait, $op);
        // impl_lhs_binary_trait!(u32, $trait, $op);
        // impl_lhs_binary_trait!(u64, $trait, $op);
        impl_lhs_binary_trait!(i8, $trait, $op);
        impl_lhs_binary_trait!(i16, $trait, $op);
        impl_lhs_binary_trait!(i32, $trait, $op);
        // impl_lhs_binary_trait!(i64, $trait, $op);
        impl_lhs_binary_trait!(f32, $trait, $op);
        // impl_lhs_binary_trait!(f16, $trait, $op);
        // impl_lhs_binary_trait!(bf16, $trait, $op);
    };
}

impl_binary_trait!(Add, add, add_with_stream);
impl_binary_trait!(Sub, sub, sub_with_stream);
impl_binary_trait!(Mul, mul, mul_with_stream);
impl_binary_trait!(Div, div, div_with_stream);

#[cfg(test)]
mod tests {
    use crate::array::MLXArray;

    #[test]
    fn test_add() {
        let array1: MLXArray = 12.0.into();
        let result = 12.0 + array1.clone() + array1 + 12.0;
        println!("{}", result.size());
        assert_eq!(1, result.size())
    }

    #[test]
    fn test_sub() {
        let array1: MLXArray = 12.0.into();
        let result = 12.0 - array1.clone() - array1 - 12.0;
        println!("{}", result.size());
        assert_eq!(1, result.size())
    }

    #[test]
    fn test_mul() {
        let array1: MLXArray = 12.0.into();
        let result = 12.0 * array1.clone() * array1 * 12.0;
        println!("{}", result.size());
        assert_eq!(1, result.size())
    }

    #[test]
    fn test_div() {
        let array1: MLXArray = 12.0.into();
        let result = 12.0 / array1.clone() / array1 / 12.0;
        println!("{}", result.size());
        assert_eq!(1, result.size())
    }
}
