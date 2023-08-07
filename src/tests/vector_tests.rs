use crate::math::*;
#[test]
fn basic_vec_mult() {
    let mut lhs = vec![0.9, 0.38, 0.79, 0.38];
    let mut rhs = vec![0.1, 0.62, 0.21, 0.62];
    vector_mult(&mut rhs, &lhs);
}
