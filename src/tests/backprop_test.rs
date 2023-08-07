use crate::math::*;
#[test]
#[should_panic]
// the expected result is the same as the output rounded.
fn delta_h_calc_test() {
    let h = vec![0.1, 0.62, 0.21, 0.62];
    let delta_o = vec![-0.33, 0.53, 0.52];
    let weight_matrix = vec![
        vec![0.48, 0.3, -0.04, 0.28],
        vec![-0.38, 0.14, -0.36, 0.44],
        vec![0.02, -0.09, -0.24, 0.27],
    ];
    let mut deriviative = h.iter().map(|v| {1. - v}).collect::<Vec<f64>>();
    vector_mult(&mut deriviative, &h);
    let transposed_weight_change = transposition(&weight_matrix);
    let res = matrix_mult(&transposed_weight_change, &vec_to_matrix(&delta_o));
    let mut delta_h = matrix_to_vec(&res);
    vector_mult(&mut delta_h, &deriviative);
    panic!("{:#?}",delta_h);
}
