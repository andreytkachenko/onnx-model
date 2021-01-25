use crate::cuda_info;
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

pub fn get_cuda_if_available(gpu_index: Option<usize>) -> InferenceDevice {
    let devices = get_inference_devices();
    let mut cpu_device: Option<InferenceDevice> = None; 
    let mut cuda_device: Option<InferenceDevice> = None; 

    for d in devices {
        match d.device_type.as_str() {
            "CPU" => { cpu_device.replace(d); },
            "CUDA" => if let Some(idx) = gpu_index {
                if idx == d.device_id.parse::<usize>().unwrap() {
                    cuda_device.replace(d);
                }
            } else {
                if cuda_device.is_none() {
                    cuda_device.replace(d);
                }
            },
            _ => ()
        }
    }

    cuda_device.or(cpu_device).unwrap()
}

pub fn get_inference_devices() -> impl Iterator<Item = InferenceDevice> {
    let mut vec: SmallVec<[InferenceDevice; 8]> = SmallVec::new();

    for p in SessionOptions::available_providers() {
        match p.as_str() {
            "CPUExecutionProvider" => {
                vec.push(InferenceDevice {
                    provider: p.as_str().into(),
                    device_id: "0".into(),
                    device_type: "CPU".into(),
                    device_name: "CPU".into(),
                });

                vec.push(InferenceDevice {
                    provider: p.as_str().into(),
                    device_id: "1".into(),
                    device_type: "CPU".into(),
                    device_name: "Parallel CPU".into(),
                });
            },
            "CUDAExecutionProvider" => {
                for i in cuda_info::list_of_cuda_devices() {
                    vec.push(InferenceDevice {
                        provider: p.as_str().into(),
                        device_id: i.index.to_string().into(),
                        device_type: "CUDA".into(),
                        device_name: i.name.into(),
                    })
                }
            },

            _ => ()
        }
    }

    vec.into_iter()
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

#[derive(Debug, Clone)]
pub struct InferenceDevice {
    provider: SmallString<[u8; 24]>,
    device_id: SmallString<[u8; 24]>,
    device_type: SmallString<[u8; 24]>,
    device_name: SmallString<[u8; 24]>,
}

pub struct OnnxInferenceModel {
    session: Session,
    run_options: RunOptions,
    input_infos: SmallVec<[TensorInfo; 4]>,
    output_infos: SmallVec<[TensorInfo; 4]>,
}

impl OnnxInferenceModel {
    pub fn new(model_filename: &str, device: InferenceDevice) -> std::result::Result<Self, Error> {
        let mut so = SessionOptions::new()?;
        let ro = RunOptions::new();

        match device.device_type.as_str() {
            "CPU" => {
                match device.device_id.as_str() {
                    "0" => { so.set_execution_mode(ExecutionMode::Sequential).unwrap(); },
                    "1" => { so.set_execution_mode(ExecutionMode::Parallel).unwrap(); },
                    _ => ()
                }

                so.add_cpu(true);
            },

            "CUDA" => {
                so.add_cuda(device.device_id.parse().unwrap());
            },

            _ => unimplemented!()
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
