mod tensor;
use tensor::Tensor;
fn main() {
    println!("Hello, world!");

    let t1 = Tensor::<f32, 2>::from_data(&[2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let t2 = Tensor::<f32, 2>::from_data(&[2, 2], vec![2.0, 2.0, 2.0, 2.0]);
    let t3 = t1 + t2;
    println!("{t3:?}")
}
