use crate::nn::NeuralNetwork;
use crate::nn_serde::*;
#[test]
fn saving_test() {
    let nn = NeuralNetwork::from_previous_save("save.json").unwrap();
    eprintln!("{nn:?}");
    panic!();
}
