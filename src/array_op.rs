use crate::array::MLXArray;
use crate::stream::{get_default_stream, MLXStream};
use crate::{mlx_add, mlx_divide, mlx_multiply, mlx_subtract};
use std::ops::{Add, Div, Mul, Sub};

impl MLXArray {
    pub fn add_with_stream(&self, rhs: impl Into<MLXArray>, stream: MLXStream) -> Self {
        let handle = unsafe { mlx_add(self.as_ptr(), rhs.into().as_ptr(), stream.as_ptr()) };
        Self::from_raw(handle)
    }

    pub fn sub_with_stream(&self, rhs: impl Into<MLXArray>, stream: MLXStream) -> Self {
        let handle = unsafe { mlx_subtract(self.as_ptr(), rhs.into().as_ptr(), stream.as_ptr()) };
        Self::from_raw(handle)
    }

    pub fn div_with_stream(&self, rhs: impl Into<MLXArray>, stream: MLXStream) -> Self {
        let handle = unsafe { mlx_divide(self.as_ptr(), rhs.into().as_ptr(), stream.as_ptr()) };
        Self::from_raw(handle)
    }

    pub fn mul_with_stream(&self, rhs: impl Into<MLXArray>, stream: MLXStream) -> Self {
        let handle = unsafe { mlx_multiply(self.as_ptr(), rhs.into().as_ptr(), stream.as_ptr()) };
        Self::from_raw(handle)
    }
}

macro_rules! impl_lhs_binary_trait {
    ($type:ty, $trait:ident, $op:ident) => {
        impl $trait<MLXArray> for $type {
            type Output = MLXArray;

            fn $op(self, rhs: MLXArray) -> Self::Output {
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
                self.$method(rhs.into(), stream)
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
