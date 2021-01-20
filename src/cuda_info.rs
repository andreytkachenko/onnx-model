use arrayvec::ArrayVec;
use std::fmt;


#[derive(Debug, Clone)]
pub (crate) struct CudaDevice {
    pub index: usize,
    pub name: String,
}

impl fmt::Display for CudaDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cuda({}): {}", self.index, self.name)?;
        Ok(())
    }
}

#[cfg(cuda)]
pub (crate) fn list_of_cuda_devices() -> ArrayVec<[CudaDevice; 8]> {
    use rustacuda::error::CudaError;
    use rustacuda::prelude::*;

    rustacuda::init(rustacuda::CudaFlags::empty()).unwrap();

    let mut res = ArrayVec::new();
    if let Ok(devices) = Device::devices() {
        for (index, d) in devices.enumerate() {
            if let Ok(device) = d? {
                if let Ok(name) = device.name() {
                    res.push(CudaDevice {
                        index,
                        name
                    })
                }
            }
        }
    }

    res
}

#[cfg(not(cuda))]
pub (crate) fn list_of_cuda_devices() -> ArrayVec<[CudaDevice; 8]> {
    ArrayVec::new()
}