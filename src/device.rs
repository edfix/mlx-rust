use std::rc::Rc;

use crate::{
    mlx_default_device, mlx_device, mlx_device_new, mlx_device_type__MLX_CPU,
    mlx_device_type__MLX_GPU, mlx_free, mlx_set_default_device,
};

#[derive(Clone)]
pub struct MlxDevice {
    handle: Rc<InnerMlxDevice>,
}

struct InnerMlxDevice(mlx_device);

impl Drop for InnerMlxDevice {
    fn drop(&mut self) {
        unsafe {
            mlx_free(self.0 as *mut std::ffi::c_void);
        }
    }
}

impl MlxDevice {
    pub fn new(device_type: DeviceType, index: i32) -> Self {
        let device_type = match device_type {
            DeviceType::CPU => mlx_device_type__MLX_CPU,
            DeviceType::GPU => mlx_device_type__MLX_GPU,
        };
        let handle = unsafe { mlx_device_new(device_type, index) };
        Self::from_raw(handle)
    }

    #[inline]
    pub fn set_default(device: MlxDevice) {
        unsafe {
            mlx_set_default_device(device.as_ptr());
        }
    }

    #[inline]
    pub fn gpu() -> Self {
        Self::new(DeviceType::GPU, 0)
    }

    #[inline]
    pub fn cpu() -> Self {
        Self::new(DeviceType::CPU, 0)
    }

    pub(crate) fn from_raw(handle: mlx_device) -> Self {
        Self {
            handle: Rc::new(InnerMlxDevice(handle)),
        }
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> mlx_device {
        self.handle.as_ref().0
    }
}

impl Default for MlxDevice {
    fn default() -> Self {
        let handle = unsafe { mlx_default_device() };
        Self::from_raw(handle)
    }
}

pub enum DeviceType {
    CPU,
    GPU,
}
