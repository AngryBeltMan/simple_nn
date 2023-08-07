use crate::math::transposition;
#[test]
fn is_working() {
    let input = vec![
        vec![1.,2.,3.],
        vec![4.,5.,6.],
    ];
    let expected_output = vec![
        vec![1.,4.],
        vec![2.,5.],
        vec![3.,6.],
    ];
    assert_eq!(transposition(&input),expected_output );
    assert_eq!(transposition(&expected_output),input );
}
