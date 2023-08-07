use crate::math::sigmoid;
// checks to make sure the outputs are between 0 and 1
#[test]
fn sigmoid_range_test() {
    for i in 1..1000 {
        let res = sigmoid(i as f64 / 2.2121);
        eprintln!("{res}");
        assert!((res >= 0.) && (res <= 1.));
    }
}
#[test]
fn sigmoid_zero_test() {
    let res = sigmoid(0.);
    assert!((res >= 0.) && (res <= 1.));
}

#[test]
fn sigmoid_negative_range_test() {
    for i in 1..1000 {
        let res = sigmoid(-i as f64 / 2.2121);
        eprintln!("{res}");
        assert!((res >= 0.) && (res <= 1.));
    }
}
