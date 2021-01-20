use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("BackendError: {0}")]
    BackendOrtError(#[from] onnxruntime::Error),

    #[error("Error: Unsupported Value Type: Tensors - the only supported type now!")]
    UnsupportedValueType,
}