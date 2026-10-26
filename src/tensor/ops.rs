use crate::tensor::Tensor;
use num_traits::{Float, Zero};
use std::ops::{Add, Mul, Sub};

fn binop<T: Float, const DIMS: usize>(
    a: &Tensor<T, DIMS>,
    b: &Tensor<T, DIMS>,
    op: fn(T, T) -> T,
) -> Tensor<T, DIMS> {
    assert!(
        a.shape() == b.shape(),
        "Add operation can only apply to Tensors with same shape"
    );

    Tensor {
        shape: *b.shape(),
        storage: a
            .storage
            .iter()
            .zip(b.storage.iter())
            .map(|(&a, &b)| op(a, b))
            .collect(),
    }
}

impl<T: Float, const DIMS: usize> Add for Tensor<T, DIMS> {
    type Output = Self;
    fn add(self, rhs: Tensor<T, DIMS>) -> Tensor<T, DIMS> {
        binop(&self, &rhs, |a, b| a + b)
    }
}

impl<T: Float, const DIMS: usize> Sub for Tensor<T, DIMS> {
    type Output = Self;
    fn sub(self, rhs: Tensor<T, DIMS>) -> Tensor<T, DIMS> {
        binop(&self, &rhs, |a, b| a - b)
    }
}

impl<T: Float, const DIMS: usize> Mul for Tensor<T, DIMS> {
    type Output = Self;
    fn mul(self, rhs: Tensor<T, DIMS>) -> Tensor<T, DIMS> {
        binop(&self, &rhs, |a, b| a * b)
    }
}
