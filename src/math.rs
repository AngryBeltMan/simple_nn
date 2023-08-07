use std::{f32::consts::E, vec};
// Eulars number but f64 instead of f32
static EULARNUMBER:f64 = E as f64;
// Simple matix type
// The inner vec represents the elements horizontally
type Matrix = Vec<Vec<f64>>;
// Calculates the dot product of to vectors
// If the two vectors are different sized it will panic.
#[inline]
pub fn dot_product(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    assert_eq!(vec1.len(), vec2.len());
    vec1.iter().zip(vec2).map(|f| f.0 * f.1 ).sum()
}
// Get the sigmoid given an input
// The output will be between 0 and 1 no matter the input.
#[inline]
pub fn sigmoid(x: f64 ) -> f64 {
    1. / (1. + (EULARNUMBER.powf(-x)))
}
// returns the relu given an input.
// The output will be between 0 and 1.
#[allow(non_snake_case)]
#[inline]
pub fn reLU(x: f64 ) -> f64 {
    if x > 0. { x } else { 0. }
}
#[inline]
pub fn vec_to_matrix(vec: &Vec<f64>) -> Matrix {
    vec.iter()
        .map(|val| vec![*val]).collect()
}
// only works for matrixes with a width of 1
#[inline]
pub fn matrix_to_vec(matrix: &Matrix) -> Vec<f64> {
    matrix.iter().map(|item| item[0] ).collect()
}

#[inline]
pub fn vector_mult(lhs: &mut Vec<f64>, rhs: &Vec<f64>) {
    for value in lhs.iter_mut().zip(rhs) {
        *value.0 *= *value.1;
    }
}

#[inline]
pub fn soft_max(input: Vec<f64>) -> Vec<f64> {
    let sum = input.iter().map(|node_val| EULARNUMBER.powf(*node_val) ).sum::<f64>();
    input.iter().map(|z| {
        EULARNUMBER.powf(*z) / (sum)
    }).collect()
}

#[inline]
pub fn transposition(matrix: &Matrix) -> Matrix {
    (0..matrix[0].len()) .map(|column| matrix.iter().map(|row| row[column]).collect() ).collect()
}

#[inline]
pub fn matrix_mult(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    lhs.iter().map(|row| {
        transposition(&rhs).iter().map(|rhs_row| {
            dot_product(row, rhs_row)
        }).collect()
    }).collect()
}

#[inline]
pub fn matrix_addition(lhs: &mut Matrix, rhs: &Matrix) {
    lhs.iter_mut()
        .zip(rhs)
        .map(|row| {
            let _ = row.0.iter_mut()
                .zip(row.1)
                .map(|value| *value.0 += *value.1)
                .collect::<()>();
        }).collect::<()>()

}

#[inline]
pub fn element_wise_multiplication(lhs: &Vec<f64>, rhs: &Vec<f64>) -> Matrix {
    lhs.iter().map(|l_value| {
        rhs.iter().map(|r_value| *r_value * (*l_value)).collect()
    }).collect()
}

#[inline]
pub fn scalar_multiplication(matrix: &mut Matrix, scalar: f64) {
    for row in matrix {
        for value in row {
            *value *= scalar;
        }
    }
}

pub trait GetMax {
    type Output;
    fn maximum(&self) -> Self::Output;
}
impl GetMax for Vec<f64> {
    type Output = usize;
    #[inline]
    fn maximum(&self) -> Self::Output {
        let mut max = f64::MIN;
        let mut max_index = 0;
        for i in 0..self.len() {
            if self[i] > max {
                max = self[i];
                max_index = i;
            }
        }
        max_index
    }
}
pub trait VecSubtraction {
    type Input;
    fn subtract(&self, other_vec: &Self::Input) -> Self;
}
impl VecSubtraction for Vec<f64> {
    type Input = Vec<f64>;
    fn subtract(&self, other_vec: &Self::Input) -> Self {
        self.iter().zip(other_vec)
            .map(|v| v.0 - v.1 )
            .collect()
    }
}
