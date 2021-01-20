mod cuda_info;
pub mod error;

use cuda_info::list_of_cuda_devices;

use onnxruntime::{
    Tensor, TensorView, Session, SessionOptions, 
    RunOptions, Env, LoggingLevel, Arguments, 
    SymbolicDim, ExecutionMode, Val
};

use ndarray::prelude::*;
use error::Error;
use lazy_static::lazy_static;
use smallvec::SmallVec;
use std::ffi::CString;

lazy_static! {
    static ref ORT_ENV: Env = Env::new(LoggingLevel::Fatal, "onnx-model").unwrap();
}

pub const MODEL_DYNAMIC_INPUT_DIMENSION: i64 = -1;

#[derive(Debug, Copy, Clone)]
pub enum OnnxInferenceDevice {
    Cpu(bool),
    Cuda(i32),
    TensorRT(i32),
}

pub fn get_cuda_if_available(device_index: Option<usize>) -> OnnxInferenceDevice {
    let devices = list_of_cuda_devices();
    let device_index = device_index.unwrap_or(0);
    
    if let Some(d) = devices.get(device_index) {
        println!("Using CUDA[{}]: {}", d.index, d.name);

        OnnxInferenceDevice::Cuda(d.index as _)
    } else {
        println!("Using CPU");

        OnnxInferenceDevice::Cpu(true)
    }
}

pub struct TensorShapeInfo {
    pub dims: SmallVec<[i64; 4]>,
    pub names: SmallVec<[Option<String>; 4]>,
}

pub struct TensorInfo {
    pub name: CString,
    pub shape: TensorShapeInfo,
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
            so.add_cuda(device);
        } else {
            unimplemented!("only cuda and cpu implemented now!");
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

    pub fn run(&mut self, inputs: &[ArrayViewD<'_, f32>]) -> std::result::Result<SmallVec<[ArrayD<f32>; 4]>, Error> {
        let mut in_vals: SmallVec<[TensorView<'_, f32>; 8]> = SmallVec::new();
        let in_names: SmallVec<[_; 8]> = self.input_infos.iter().map(|i|i.name.as_c_str()).collect();
        let out_names: SmallVec<[_; 8]> = self.output_infos.iter().map(|i|i.name.as_c_str()).collect();

        for i in inputs {
            in_vals.push(TensorView::new(i.shape(), i.as_slice().unwrap()));
        }

        let in_vals_refs: SmallVec<[&Val; 8]> = in_vals.iter().map(|x|x.as_ref()).collect();

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
