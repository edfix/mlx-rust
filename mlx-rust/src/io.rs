use std::{ffi::CString, os::unix::ffi::OsStrExt, path::Path};

use mlx_sys::{fclose, FILE, fopen, mlx_load_safetensors, mlx_map_string_to_array, mlx_map_string_to_array_, mlx_map_string_to_array_get, mlx_map_string_to_array_iterate, mlx_map_string_to_array_iterator, mlx_map_string_to_array_iterator_, mlx_map_string_to_array_iterator_end, mlx_map_string_to_array_iterator_key, mlx_map_string_to_array_iterator_next, mlx_map_string_to_array_iterator_value, mlx_map_string_to_string, mlx_map_string_to_string_, mlx_map_string_to_string_get, mlx_map_string_to_string_iterate, mlx_map_string_to_string_iterator, mlx_map_string_to_string_iterator_, mlx_map_string_to_string_iterator_end, mlx_map_string_to_string_iterator_key, mlx_map_string_to_string_iterator_next, mlx_map_string_to_string_iterator_value, mlx_safetensors, mlx_safetensors_, mlx_safetensors_data, mlx_safetensors_metadata};

use crate::{MLXArray, object::MLXObject, stream::MLXStream, string::MLXString};

pub struct SafeTensors {
    handle: MLXObject<mlx_safetensors_>,
}

pub struct SafeTensorMetadata(MLXObject<mlx_map_string_to_string_>);
impl SafeTensorMetadata {
    fn from_raw(handle: mlx_map_string_to_string) -> Self {
        Self(MLXObject::from_raw(handle))
    }

    pub fn get(&self, key: &str) -> Option<MLXString> {
        let key = MLXString::new(key);
        let handle = unsafe { mlx_map_string_to_string_get(self.0.as_ptr(), key.as_ptr()) };
        if handle.is_null() {
            None
        } else {
            Some(MLXString::from_raw(handle))
        }
    }
}

impl IntoIterator for SafeTensorMetadata {
    type Item = (String, String);

    type IntoIter = SafeTensorMetadataIterator;

    fn into_iter(self) -> Self::IntoIter {
        let handle = unsafe { mlx_map_string_to_string_iterate(self.0.as_ptr()) };
        SafeTensorMetadataIterator(MLXObject::from_raw(handle))
    }
}

pub struct SafeTensorMetadataIterator(MLXObject<mlx_map_string_to_string_iterator_>);

impl SafeTensorMetadataIterator {
    pub fn as_ptr(&self) -> mlx_map_string_to_string_iterator {
        self.0.as_ptr()
    }
}

impl Iterator for SafeTensorMetadataIterator {
    type Item = (String, String);

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if !mlx_map_string_to_string_iterator_end(self.as_ptr()) {
                let key = mlx_map_string_to_string_iterator_key(self.as_ptr());
                let key = MLXString::from_raw(key);
                let value = mlx_map_string_to_string_iterator_value(self.as_ptr());
                let value = MLXString::from_raw(value);
                mlx_map_string_to_string_iterator_next(self.as_ptr());
                Some((key.to_string().unwrap(), value.to_string().unwrap()))
            } else {
                None
            }
        }
    }
}
pub struct SafeTensorData(MLXObject<mlx_map_string_to_array_>);

impl SafeTensorData {
    fn from_raw(handle: mlx_map_string_to_array) -> Self {
        Self(MLXObject::from_raw(handle))
    }

    pub fn get(&self, key: &str) -> Option<MLXArray> {
        let key = MLXString::new(key);
        let handle = unsafe { mlx_map_string_to_array_get(self.0.as_ptr(), key.as_ptr()) };
        if handle.is_null() {
            None
        } else {
            Some(MLXArray::from_raw(handle))
        }
    }
}

impl IntoIterator for SafeTensorData {
    type Item = (String, MLXArray);

    type IntoIter = SafeTensorMetadataDataIterator;

    fn into_iter(self) -> Self::IntoIter {
        let handle = unsafe { mlx_map_string_to_array_iterate(self.0.as_ptr()) };
        SafeTensorMetadataDataIterator(MLXObject::from_raw(handle))
    }
}

pub struct SafeTensorMetadataDataIterator(MLXObject<mlx_map_string_to_array_iterator_>);

impl SafeTensorMetadataDataIterator {
    pub fn as_ptr(&self) -> mlx_map_string_to_array_iterator {
        self.0.as_ptr()
    }
}

impl Iterator for SafeTensorMetadataDataIterator {
    type Item = (String, MLXArray);

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if !mlx_map_string_to_array_iterator_end(self.as_ptr()) {
                let key = mlx_map_string_to_array_iterator_key(self.as_ptr());
                let key = MLXString::from_raw(key);
                let value = mlx_map_string_to_array_iterator_value(self.as_ptr());
                let value = MLXArray::from_raw(value);
                mlx_map_string_to_array_iterator_next(self.as_ptr());
                Some((key.to_string().unwrap(), value))
            } else {
                None
            }
        }
    }
}

struct CFile(*mut FILE);

impl CFile {
    fn open(path: &Path) -> Self {
        let path_bytes = path.as_os_str().as_bytes();

        // Attempt to create a CString
        let file_path = CString::new(path_bytes).unwrap().into_raw();
        let file = unsafe { fopen(file_path, c"rb".as_ptr()) };
        CFile(file)
    }

    fn as_raw(&self) -> *mut FILE {
        self.0
    }
}

impl Drop for CFile {
    fn drop(&mut self) {
        unsafe { fclose(self.0) };
    }
}

impl SafeTensors {
    pub fn new(path: &str, stream: MLXStream) -> Self {
        // let file = CFile::open(path);
        let path = MLXString::new(path);
        let handle = unsafe { mlx_load_safetensors(path.as_ptr(), stream.as_ptr()) };
        SafeTensors::from_raw(handle)
    }

    fn from_raw(handle: mlx_safetensors) -> Self {
        Self {
            handle: MLXObject::from_raw(handle),
        }
    }

    fn as_ptr(&self) -> mlx_safetensors {
        self.handle.as_ptr()
    }
}

impl SafeTensors {
    pub fn metadata(&self) -> SafeTensorMetadata {
        let handle = unsafe { mlx_safetensors_metadata(self.as_ptr()) };
        SafeTensorMetadata::from_raw(handle)
    }

    pub fn data(&self) -> SafeTensorData {
        let handle = unsafe { mlx_safetensors_data(self.as_ptr()) };
        SafeTensorData::from_raw(handle)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs, slice::from_raw_parts};

    use safetensors::{Dtype, serialize, tensor::TensorView};

    use crate::stream::get_default_stream;

    use super::SafeTensors;

    #[test]
    fn test_load_safetensor() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let shape = vec![1, 2, 3];
        let attn_0 = TensorView::new(Dtype::F32, shape, &data).unwrap();
        let metadata: HashMap<String, TensorView> =
            [("a".to_string(), attn_0)].into_iter().collect();

        let out = serialize(&metadata, &None).unwrap();

        let dir = std::env::temp_dir();
        let file_path = dir.join("test.safetensors");
        let file_path = file_path.as_path();
        fs::write(file_path, &out).unwrap();

        let data = fs::read(file_path).unwrap();
        let raw_st = safetensors::SafeTensors::deserialize(&data).unwrap();
        let st = SafeTensors::new(file_path.to_str().unwrap(), get_default_stream());

        for (i, (key, value)) in st.data().into_iter().enumerate() {
            let raw_data = raw_st.tensor(&key).unwrap();
            println!("raw_data: {:?}", convert_slice::<f32>(raw_data.data()));
            assert_eq!(convert_slice::<f32>(raw_data.data()), value.to_slice::<f32>().unwrap());
            println!("{} {}: {}, shape{:?}", i, key, value, value.shape());
        }
    }

    fn convert_slice<T: Clone>(data: &[u8]) -> Vec<T> {
        let size_in_bytes = std::mem::size_of::<T>();
        let elem_count = data.len() / size_in_bytes;
        if (data.as_ptr() as usize) % size_in_bytes == 0 {
            // SAFETY: This is safe because we just checked that this
            // was correctly aligned.
            let data: &[T] = unsafe { from_raw_parts(data.as_ptr() as *const T, elem_count) };
            data.to_vec()
        } else {
            // XXX: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
            // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
            let mut c: Vec<T> = Vec::with_capacity(elem_count);
            // SAFETY: We just created c, so the allocated memory is necessarily
            // contiguous and non-overlapping with the view's data.
            // We're downgrading the `c` pointer from T to u8, which removes alignment
            // constraints.
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
                c.set_len(elem_count)
            }
            c
        }
    }
}
