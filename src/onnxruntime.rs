use crate::cuda_info::{CudaDevice, list_of_cuda_devices};
use crate::error::Error;

use onnxruntime::{Arguments, Env, ExecutionMode, LoggingLevel, RunOptions, Session, SessionOptions, SymbolicDim, Tensor, TensorView, Val};

use ndarray::prelude::*;
use lazy_static::lazy_static;
use smallvec::SmallVec;
use smallstr::SmallString;

use core::fmt;
use std::ffi::CString;

lazy_static! {
    static ref ORT_ENV: Env = Env::new(LoggingLevel::Error, "onnx-model").unwrap();
}

pub const MODEL_DYNAMIC_INPUT_DIMENSION: i64 = -1;

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum OnnxInferenceDevice {
    Cpu(bool),
    Cuda(CudaDevice),
    TensorRT(CudaDevice),
}

pub fn get_cuda_if_available(device_index: Option<usize>) -> OnnxInferenceDevice {
    let devices = list_of_cuda_devices();
    let device_index = device_index.unwrap_or(0);
    
    if let Some(d) = devices.into_iter().nth(device_index) {
        if cfg!(feature = "tensorrt") {
            OnnxInferenceDevice::TensorRT(d)
        } else if cfg!(feature = "cuda") {
            OnnxInferenceDevice::Cuda(d)
        } else {
            OnnxInferenceDevice::Cpu(true)
        }
    } else {
        OnnxInferenceDevice::Cpu(true)
    }
}

#[derive(Clone)]
pub struct TensorShapeInfo {
    pub dims: SmallVec<[i64; 4]>,
    pub names: SmallVec<[Option<String>; 4]>,
}

impl fmt::Display for TensorShapeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (idx, (dim, name)) in self.dims.iter().zip(self.names.iter()).enumerate() {
            if idx != 0 {
                write!(f, ", ")?;
            }

            if let Some(x) = name {
                write!(f, "{}", x)?;
            } else {
                write!(f, "{}", dim)?;
            }
        }

        write!(f, ")")?;
        
        Ok(())
    }
}

impl fmt::Debug for TensorShapeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self, f)
    }
}

#[derive(Clone)]
pub struct TensorInfo {
    pub name: CString,
    pub shape: TensorShapeInfo,
}

impl fmt::Display for TensorInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor[{:?}]({})", self.name, self.shape)?;
        Ok(())
    }
}

impl fmt::Debug for TensorInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor[{:?}]({})", self.name, self.shape)?;
        Ok(())
    }
}

pub struct InferenceDevice {
    name: SmallString<[u8; 32]>,
    index: usize
}

pub struct OnnxInferenceModel {
    session: Session,
    run_options: RunOptions,
    input_infos: SmallVec<[TensorInfo; 4]>,
    output_infos: SmallVec<[TensorInfo; 4]>,
}

impl OnnxInferenceModel {
    pub fn new(model_filename: &str, device: OnnxInferenceDevice) -> std::result::Result<Self, Error> {
        let mut so = SessionOptions::new()?;
        let ro = RunOptions::new();

        if let OnnxInferenceDevice::Cpu(is_parallel) = device {
            if is_parallel {
                so.set_execution_mode(ExecutionMode::Parallel).unwrap();
            }

            so.add_cpu(true);
        } else if let OnnxInferenceDevice::Cuda(device) = device {
            so.add_cuda(device.index as _);
        } else if let OnnxInferenceDevice::TensorRT(_device) = device {
            // so.add_tensorrt(device.index as _);
        } else {
            unimplemented!("only cuda, tensorrt and cpu implemented now!");
        }

        let session = Session::new(&ORT_ENV, model_filename, &so).unwrap();
        let input_infos = Self::tensor_info(session.inputs());
        let output_infos = Self::tensor_info(session.outputs());

        Ok(Self {
            session,
            input_infos,
            output_infos,
            run_options: ro
        })
    }

    pub fn get_inference_devices(&self) -> impl Iterator<Item = InferenceDevice> {
            core::iter::once(InferenceDevice {
                name: "CPU".into(),
                index: 0
            }).chain(
                core::iter::once(InferenceDevice {
                    name: "Parallel CPU".into(),
                    index: 1
                }).chain(
                    core::iter::once(InferenceDevice {
                        name: "CUDA:0".into(),
                        index: 2
                    }).chain(core::iter::once(InferenceDevice {
                        name: "TensorRT:0".into(),
                        index: 3
                    }),
                )
            )
        )
    }

    #[inline]
    pub fn get_input_infos(&self) -> &[TensorInfo] {
        self.input_infos.as_slice()
    }

    #[inline]
    pub fn get_output_infos(&self) -> &[TensorInfo] {
        self.output_infos.as_slice()
    }

    pub fn run(&self, inputs: &[ArrayViewD<'_, f32>]) -> std::result::Result<SmallVec<[ArrayD<f32>; 4]>, Error> {
        let mut in_vals: SmallVec<[CowArray<'_, f32, IxDyn>; 8]> = SmallVec::new();
        let in_names: SmallVec<[_; 8]> = self.input_infos.iter().map(|i|i.name.as_c_str()).collect();
        let out_names: SmallVec<[_; 8]> = self.output_infos.iter().map(|i|i.name.as_c_str()).collect();

        for i in inputs {
            in_vals.push(i.as_standard_layout());
        }

        let in_vals_views: SmallVec<[_; 8]> = in_vals.iter()
            .map(|x|TensorView::new(x.shape(), x.as_slice().unwrap()))
            .collect();

        let in_vals_refs: SmallVec<[&Val; 8]> = in_vals_views.iter().map(|x| x.as_ref()).collect();

        let out_vals = self.session
            .run_raw(
                &self.run_options,
                in_names.as_slice(),
                in_vals_refs.as_slice(),
                out_names.as_slice(),
            )?;

        let mut tensors: SmallVec<[ArrayD<f32>; 4]> = SmallVec::with_capacity(out_vals.len());
        
        for v in out_vals {
            let x: Tensor<f32> = v.as_tensor()
                .map_err(|_| Error::UnsupportedValueType)?;
            
            let shape: SmallVec<[usize; 8]> = x.dims().into_iter().map(|&x|x as usize).collect();

            tensors.push(ArrayD::from_shape_vec(shape.as_slice(), x.to_vec()).unwrap())
        }

        Ok(tensors)
    }

    fn tensor_info(args: Arguments<'_>) -> SmallVec<[TensorInfo; 4]> {
        args.map(|arg| {
            if let Some(info) = arg.tensor_info() {
                let dims = info.symbolic_dims();

                let mut dim_names = SmallVec::with_capacity(dims.len());
                let mut dim_shape = SmallVec::with_capacity(dims.len());

                for dim in dims {
                    match dim {
                        SymbolicDim::Symbolic(name) => {
                            dim_names.push(name.to_str().ok().map(|s|s.to_owned()));
                            dim_shape.push(-1i64);
                        }
                        SymbolicDim::Fixed(x) => {
                            dim_names.push(None);
                            dim_shape.push(x as i64);
                        },
                    }
                }

                TensorInfo {
                    name: arg.name().as_c_str().to_owned(),
                    shape: TensorShapeInfo {
                        dims: dim_shape,
                        names: dim_names,
                    },
                }
            } else {
                panic!("Supported only tensors!");
            }
        })
        .collect()
    }
}
