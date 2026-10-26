use num_traits::{Float, Zero};
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Tensor<Dtype, const DIMS: usize> {
    pub(crate) shape: [usize; DIMS],
    pub(crate) storage: Vec<Dtype>,
}

impl<T, const DIMS: usize> Tensor<T, DIMS> {
    pub fn shape(&self) -> &[usize; DIMS] {
        &self.shape
    }

    pub fn data(&self) -> &[T] {
        &self.storage
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.storage
    }

    pub fn numel(&self) -> usize {
        self.storage.len()
    }

    fn checked_numel(shape: &[usize; DIMS]) -> usize {
        // Una dimensión vacía implica cero elementos.
        if shape.contains(&0) {
            return 0;
        }

        shape
            .iter()
            .try_fold(1usize, |size, &dim| size.checked_mul(dim))
            .expect("Tensor element count overflows usize")
    }
}

impl<T: Zero + Clone, const DIMS: usize> Tensor<T, DIMS> {
    pub fn new() -> Self {
        Self::from_shape(&[0; DIMS])
    }

    pub fn from_shape(shape: &[usize; DIMS]) -> Self {
        let size = Self::checked_numel(shape);

        Self {
            shape: *shape,
            storage: vec![T::zero(); size],
        }
    }

    pub fn from_data(shape: &[usize; DIMS], data: Vec<T>) -> Self {
        let size = Self::checked_numel(shape);

        assert_eq!(data.len(), size, "Tensor: data length does not match shape",);

        Self {
            shape: *shape,
            storage: data,
        }
    }
}

impl<T: Clone, const DIMS: usize> Tensor<T, DIMS> {
    pub fn from_tensor(t: &Self) -> Self {
        t.clone()
    }
}
