#![allow(unused_imports)]
use crate::math::dot_product;
#[test]
fn multiply_two_vectors() {
    let res = dot_product(&vec![1.,2.,3.], &vec![1.,5.,7.]);
    assert_eq!(res, 32.);
}
#[test]
#[should_panic]
fn multiply_two_different_sized_vecs() {
    dot_product(&vec![1.,2.,3.], &vec![1.,5.]);
}
