use crate::math::VecSubtraction;
#[test]
fn vec_subtraction() {
    let lhs = vec![1.,2.,3.];
    let rhs = vec![3.,2.,1.];
    let res = lhs.subtract(&rhs);
    assert_eq!(res, vec![-2., 0., 2.])
}
