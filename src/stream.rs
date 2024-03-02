use crate::{device::MlxDevice, mlx_default_stream, mlx_free, mlx_stream, mlx_stream_new};
use std::{ffi::c_void, rc::Rc};

pub struct MlxStream {
    innner: Rc<InnerMlxStream>,
}

impl MlxStream {
    #[inline]
    pub fn new(dev: MlxDevice, index: i32) -> Self {
        let handle = unsafe { mlx_stream_new(index, dev.as_ptr()) };
        Self::from_raw(handle)
    }

    #[inline]
    fn from_raw(handle: mlx_stream) -> Self {
        MlxStream {
            innner: Rc::new(InnerMlxStream(handle)),
        }
    }

    pub fn default_stream(dev: MlxDevice) -> Self {
        let stream = unsafe { mlx_default_stream(dev.as_ptr()) };
        Self::from_raw(stream)
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> mlx_stream {
        self.innner.as_ref().0
    }
}

struct InnerMlxStream(mlx_stream);

impl Drop for InnerMlxStream {
    fn drop(&mut self) {
        unsafe {
            mlx_free(self.0 as *mut c_void);
        }
    }
}
