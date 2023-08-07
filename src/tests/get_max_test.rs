use crate::math::GetMax;
#[test]
fn get_max_test() {
    let array: Vec<f64> = vec![0.1,0.2,0.5,0.3];
    let max = array.maximum();
    assert_eq!(max, 2)
}
