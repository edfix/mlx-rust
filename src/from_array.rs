use crate::{array::MLXArray, r#type::ScalarMlxType};
use core::slice;

impl MLXArray {
    pub fn to_scalar<T: ScalarMlxType>(&self) -> Result<T, ()> {
        assert!(<T as ScalarMlxType>::mlx_array_dtype == self.dtype());
        self.eval();
        let r = unsafe { <T as ScalarMlxType>::to_scalar(self.as_ptr()) };
        Ok(r)
    }

    pub fn to_slice<T: ScalarMlxType>(&self) -> Result<&[T], ()> {
        assert!(<T as ScalarMlxType>::mlx_array_dtype == self.dtype());
        self.eval();
        unsafe {
            let ptr = <T as ScalarMlxType>::to_slice(self.as_ptr());
            let len = self.size();
            if ptr.is_null() || len == 0 {
                return Err(());
            }
            Ok(slice::from_raw_parts(ptr, len))
        }
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};

    use crate::array::MLXArray;

    #[test]
    fn test_to_scalar() {
        let array1: MLXArray = 2.into();
        assert_eq!(2, array1.to_scalar().unwrap())
    }

    #[test]
    fn test_as_float32_slice() {
        let array1 = MLXArray::array(&[123., 134.], &[2]) + 1;
        let r: &[f32] = array1.to_slice().unwrap();
        assert_eq!(vec![124.0, 135.0], r.to_vec())
    }

    #[test]
    fn test_as_float16_slice() {
        let array1 = MLXArray::array(&[f16::from_f32(123.0)], &[1]) + 1;
        let r: &[f16] = array1.to_slice().unwrap();
        assert_eq!(vec![f16::from_f32(124.0)], r.to_vec())
    }

    #[test]
    fn test_as_bfloat16_slice() {
        let array1 = MLXArray::array(&[bf16::from_f32(123.0)], &[1]) + 1;
        let r: &[bf16] = array1.to_slice().unwrap();
        assert_eq!(vec![bf16::from_f32(124.0)], r.to_vec())
    }
}
