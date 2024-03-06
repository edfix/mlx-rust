use mlx_sys::{
    mlx_array, mlx_vector_array, mlx_vector_array_, mlx_vector_array_add,
    mlx_vector_array_add_arrays, mlx_vector_array_from_array, mlx_vector_array_from_arrays,
    mlx_vector_array_get, mlx_vector_array_new, mlx_vector_array_size,
};

use crate::{object::MLXObject, MLXArray};

/// A vector of MLX arrays.
///
/// The `VectorMLXArray` type provides a safe and convenient way to work with a collection
/// of `MLXArray` objects in the MLX library. It encapsulates an `mlx_vector_array` object
/// and manages its lifetime using reference counting.
///
/// `VectorMLXArray` can be created using the `new` method, which returns an empty vector.
/// Arrays can be added to the vector using the `add` and `add_arrays` methods. The `get`
/// method allows retrieving an array at a specific index, while the `len` method returns
/// the number of arrays in the vector.
///
/// # Examples
///
/// ```
/// use mlx_rust::{VectorMLXArray, MLXArray};
/// let mut vec = VectorMLXArray::new();
/// let arr1 = 12.0.into();
/// let arr2 = 13.0.into();
/// vec.add(arr1);
/// vec.add(arr2);
///
/// let arr = vec.get(0);
/// let size = vec.len();
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct VectorMLXArray {
    inner: MLXObject<mlx_vector_array_>,
}

impl VectorMLXArray {
    /// Creates a new empty vector of arrays.
    pub fn new() -> Self {
        Self::from_raw(unsafe { mlx_vector_array_new() })
    }

    pub fn from_raw(handle: mlx_vector_array) -> Self {
        Self {
            inner: MLXObject::from_raw(handle),
        }
    }

    pub fn as_ptr(&self) -> mlx_vector_array {
        self.inner.as_ptr()
    }

    /// Creates a new vector of arrays containing a single specified array.
    pub fn from_array(arr: MLXArray) -> Self {
        //safety The reference count of the given array will be increased.
        let handle = unsafe { mlx_vector_array_from_array(arr.as_ptr()) };
        Self::from_raw(handle)
    }

    /// Adds an array to the vector of arrays.
    ///
    /// The reference count of the given array will be increased.
    pub fn add(&mut self, arr: MLXArray) {
        unsafe {
            // mlx_retain(arr.as_ptr() as *mut ::std::os::raw::c_void);
            mlx_vector_array_add(self.as_ptr(), arr.as_ptr())
        }
    }

    /// Adds several arrays to the vector of arrays.
    pub fn add_arrays(&mut self, arrs: Vec<MLXArray>) {
        let ptr = arrs.as_ptr() as *const mlx_array;
        let num_arrs = arrs.len();

        unsafe {
            //safety
            //The reference count of the given arrays will be increased.
            mlx_vector_array_add_arrays(self.inner.as_ptr(), ptr, num_arrs)
        }
    }

    /// Retrieves the array at the specified index in the vector of arrays.
    pub fn get(&self, index: usize) -> Option<MLXArray> {
        unsafe {
            let ptr = mlx_vector_array_get(self.as_ptr(), index);
            if ptr.is_null() {
                None
            } else {
                let r = MLXArray::from_raw(ptr);
                Some(r)
            }
        }
    }

    /// Returns the number of arrays in the vector of arrays.
    pub fn len(&self) -> usize {
        unsafe { mlx_vector_array_size(self.as_ptr()) }
    }
}

impl<T: Into<MLXArray>> From<(T, T)> for VectorMLXArray {
    fn from(value: (T, T)) -> Self {
        let array: [MLXArray; 2] = [value.0.into(), value.1.into()];
        let ptr = array.as_ptr() as *mut mlx_array;
        let num = 2;
        //safety: The reference count of the given arrays will be increased.
        let handle = unsafe { mlx_vector_array_from_arrays(ptr, num) };
        VectorMLXArray::from_raw(handle)
    }
}

impl<T: Into<MLXArray>> From<T> for VectorMLXArray {
    fn from(value: T) -> Self {
        let array: [MLXArray; 1] = [value.into()];
        let ptr = array.as_ptr() as *mut mlx_array;
        let num = 1;
        //safety: The reference count of the given arrays will be increased.
        let handle = unsafe { mlx_vector_array_from_arrays(ptr, num) };
        VectorMLXArray::from_raw(handle)
    }
}

// impl<T: AsRef<[MLXArray]>> From<T> for VectorMLXArray {
//     fn from(arrs: T) -> Self {
//         let a: &[MLXArray] = arrs.as_ref();
//         let ptr = a.as_ptr() as *mut mlx_array;
//         let num_arrs = a.len();

//         //safety: The reference count of the given arrays will be increased.
//         let handle = unsafe { mlx_vector_array_from_arrays(ptr, num_arrs) };
//         VectorMLXArray::from_raw(handle)
//     }
// }

#[cfg(test)]
mod tests {
    use crate::array::MLXArray;

    use super::VectorMLXArray;

    #[test]
    fn test_get_vector_array() {
        let array: MLXArray = 12.0.into();
        for _ in 0..1000 {
            let r = {
                let mut vec = VectorMLXArray::new();
                vec.add(array.clone());
                assert_eq!(vec.len(), 1);
                let first_get_result = vec.get(0).unwrap();
                first_get_result
            };

            assert_eq!(array.to_string(), r.to_string())
        }
    }

    #[test]
    fn test_vector_array_size() {
        let array: MLXArray = 12.0.into();
        let mut vec = VectorMLXArray::new();
        vec.add(array.clone());
        assert_eq!(vec.len(), 1);
    }

    #[test]
    fn test_vector_array_from_array() {
        let array: MLXArray = 12.0.into();
        let vec = VectorMLXArray::from_array(array.clone());
        assert_eq!(vec.len(), 1);
        assert_eq!(array.to_string(), vec.get(0).unwrap().to_string());
    }

    #[test]
    fn test_vector_array_from_vector() {
        let array: MLXArray = 12.0.into();
        let vec: VectorMLXArray = (array.clone(), array.clone()).into();
        assert_eq!(vec.len(), 2);
        assert_eq!(array.to_string(), vec.get(0).unwrap().to_string());
    }

    #[test]
    fn test_vector_array_from_rust_array() {
        let array: MLXArray = 12.0.into();
        let vec: VectorMLXArray = (array.clone(), array.clone()).into();
        assert_eq!(vec.len(), 2);
        assert_eq!(array.to_string(), vec.get(0).unwrap().to_string());
    }
}
