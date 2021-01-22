use ndarray_stats::QuantileExt;

fn main() {
    let arg = std::env::args().nth(1).unwrap();
    let device = onnx_model::get_cuda_if_available();
    println!("Using: {:?}", device);

    let model = onnx_model::OnnxInferenceModel::new("./examples/models/resnet18-v2-7.onnx", device).unwrap();

    println!("inputs: {:?}", model.get_input_infos());
    println!("outputs: {:?}", model.get_output_infos());

    let img = image::open(arg).unwrap();
    let img = img.resize_to_fill(224, 224, image::imageops::FilterType::CatmullRom).to_rgb8().into_vec();
    let arr = ndarray::Array4::from_shape_vec([1, 224, 224, 3], img).unwrap();
    let arr = arr.permuted_axes([0, 3, 1, 2]).mapv(|x| x as f32 / 255.0);
    let res = model.run(&[arr.into_dyn().view()]).unwrap();

    println!("{:?}", res.into_iter().next().unwrap().into_dimensionality::<ndarray::Ix2>().unwrap().argmax().unwrap());
}