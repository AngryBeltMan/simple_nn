use crate::math::*;
#[test]
#[should_panic]
// the output is the same as the answer just not rounded
fn multiply_two_vectors() {
    let lhs = vec![
        vec![-0.33],
        vec![0.53],
        vec![0.52],
    ];
    let rhs = vec![
        vec![0.1,0.62,0.21,0.62],
    ];
    let answer = vec![
        vec![-0.03,-0.2,-0.07,-0.2],
        vec![0.05,0.33,0.11,0.33],
        vec![0.05, 0.32, 0.11, 0.32],
    ];
    assert_eq!(matrix_mult(&lhs, &rhs), answer);
}
#[test]
fn multiply_by_a_scalar() {
    let mut matrix = vec![
        vec![1.,2.,3.],
        vec![4.,5.,6.],
    ];
    scalar_multiplication(&mut matrix, 2.0);
    let res = vec![
        vec![2.,4.,6.],
        vec![8.,10.,12.],
    ];
    assert_eq!(matrix, res);
}
#[test]
fn backprop_test() {
    let delta_o = vec![-0.33,0.53, 0.52];
    let input = vec![0.1, 0.62, 0.21, 0.62];
    let learn_rate = -0.01;
    let mut w_h_o = element_wise_multiplication(&delta_o, &input);
    scalar_multiplication(&mut w_h_o, learn_rate);
    // panic!("{:#?}", w_h_o);
}
#[test]
fn element_wise_multiplication_test() {
    let res = element_wise_multiplication(&vec![1.,2.], &vec![3.,4.,5.]);
    let answer = vec![
        vec![3.,4.,5.],
        vec![6.,8.,10.],
    ];
    assert_eq!(res, answer)
}

