use std::{fmt::Display, sync::Mutex};

use crate::{
    device::{get_default_device, MLXDevice},
    mlx_default_stream, mlx_stream, mlx_stream_, mlx_stream_new,
    object::MLXObject,
};

lazy_static::lazy_static! {
    static ref DEFAULT_STREAM: Mutex<MLXStream> = Mutex::new(MLXStream::from_raw(unsafe {
        mlx_default_stream(get_default_device().as_ptr())
    }));
}

pub fn get_default_stream() -> MLXStream {
    let g = DEFAULT_STREAM.lock().unwrap();
    g.clone()
}

#[derive(Clone, PartialEq, Debug)]
pub struct MLXStream {
    innner: MLXObject<mlx_stream_>,
}

impl Display for MLXStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.innner.fmt(f)
    }
}

impl MLXStream {
    #[inline]
    pub fn new(dev: MLXDevice, index: i32) -> Self {
        let handle = unsafe { mlx_stream_new(index, dev.as_ptr()) };
        Self::from_raw(handle)
    }

    #[inline]
    fn from_raw(handle: mlx_stream) -> Self {
        MLXStream {
            innner: MLXObject::from_raw(handle),
        }
    }

    pub fn default_stream(dev: MLXDevice) -> Self {
        let stream = unsafe { mlx_default_stream(dev.as_ptr()) };
        Self::from_raw(stream)
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> mlx_stream {
        self.innner.as_ptr()
    }
}
