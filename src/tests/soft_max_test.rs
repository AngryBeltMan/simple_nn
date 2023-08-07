use crate::math::soft_max;

// checks to see if the soft max function normalizes all of the values
#[test]
fn is_normalized() {
    let res = soft_max(vec![1.3, 5.1, 2.2, 0.7, 1.1]);
    assert_eq!(res.iter().sum::<f64>(), 1.);
}
