use std::fmt;

#[derive(Clone)]
pub struct CudaDevice {
    pub index: usize,
    pub name: String,
}

impl fmt::Display for CudaDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{{}: {}}}", self.index, self.name)?;
        Ok(())
    }
}

impl fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{{}: {}}}", self.index, self.name)?;
        Ok(())
    }
}

#[cfg(any(feature = "cuda", feature = "tensorrt"))]
pub(crate) fn list_of_cuda_devices() -> ArrayVec<[CudaDevice; 8]> {
    use rustacuda::prelude::*;
    let mut res = ArrayVec::new();

    if rustacuda::init(rustacuda::CudaFlags::empty()).is_ok() {
        if let Ok(devices) = Device::devices() {
            for (index, d) in devices.enumerate() {
                if let Ok(device) = d {
                    if let Ok(name) = device.name() {
                        res.push(CudaDevice { index, name })
                    }
                }
            }
        }
    }

    res
}
