use std::{fmt::Display, sync::Mutex};

use crate::{
    mlx_default_device, mlx_device, mlx_device_, mlx_device_get_type, mlx_device_new,
    mlx_device_type__MLX_CPU, mlx_device_type__MLX_GPU, object::MLXObject,
};

lazy_static::lazy_static! {
    static ref DEFAULT_DEVICE: Mutex<MLXDevice> = Mutex::new(MLXDevice::from_raw(unsafe {
        mlx_default_device()
    }));
}

pub fn get_default_device() -> MLXDevice {
    let guard = DEFAULT_DEVICE.lock().unwrap();
    // let handle = unsafe { mlx_default_device() };
    // MLXDevice::from_raw(handle)
    guard.clone()
}

//todo is it safe?
pub fn set_default_device(device: MLXDevice) {
    let mut guard = DEFAULT_DEVICE.lock().unwrap();
    // unsafe {
    //     mlx_set_default_device(device.as_ptr());
    //     println!("after get: {:?} ", mlx_default_device())
    // }
    *guard = device
}

#[derive(Clone, PartialEq, Debug)]
pub struct MLXDevice {
    handle: MLXObject<mlx_device_>,
}

impl Display for MLXDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.handle.fmt(f)
    }
}

impl MLXDevice {
    pub fn new(device_type: DeviceType, index: i32) -> Self {
        let device_type = match device_type {
            DeviceType::CPU => mlx_device_type__MLX_CPU,
            DeviceType::GPU => mlx_device_type__MLX_GPU,
        };
        let handle = unsafe { mlx_device_new(device_type, index) };
        Self::from_raw(handle)
    }

    #[inline]
    pub fn gpu() -> Self {
        Self::new(DeviceType::GPU, 0)
    }

    #[inline]
    pub fn cpu() -> Self {
        Self::new(DeviceType::CPU, 0)
    }

    pub fn device_type(&self) -> Result<DeviceType, DeviceError> {
        let handle = unsafe { mlx_device_get_type(self.as_ptr()) };
        handle.try_into()
    }

    pub(crate) fn from_raw(handle: mlx_device) -> Self {
        Self {
            handle: MLXObject::from_raw(handle),
        }
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> mlx_device {
        self.handle.as_ptr()
    }
}

#[repr(C)]
#[derive(Debug, PartialEq)]
pub enum DeviceType {
    CPU = 0,
    GPU = 1,
}

#[derive(Debug)]
pub enum DeviceError {
    UnknowError(u32),
}

impl TryFrom<u32> for DeviceType {
    type Error = DeviceError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(DeviceType::CPU),
            1 => Ok(DeviceType::GPU),
            _ => Err(DeviceError::UnknowError(value)), // Return an error if the value doesn't match any variant
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::device::DeviceType;

    use super::{get_default_device, set_default_device, MLXDevice};

    #[test]
    pub fn test_get_default_device() {
        let device_type = get_default_device().device_type().unwrap();
        assert_eq!(DeviceType::GPU, device_type)
    }

    #[test]
    pub fn test_set_default_device() {
        let cpu = MLXDevice::cpu();
        println!("before set: {:?}", get_default_device());
        set_default_device(cpu.clone());
        assert_eq!(cpu, get_default_device())
    }
}
