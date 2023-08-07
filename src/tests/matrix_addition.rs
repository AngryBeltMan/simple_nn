use crate::math::*;
#[test]
fn addition_test() {
    let mut lhs = vec![
        vec![1., 2.,3.],
        vec![3., 2.,1.],
    ];
    let rhs = vec![
        vec![2., 2.,2.],
        vec![3., 2.,1.],
    ];
    matrix_addition(&mut lhs, &rhs);
    let res = vec![
        vec![3., 4.,5.],
        vec![6., 4.,2.],
    ];
    assert_eq!(lhs, res);
}
