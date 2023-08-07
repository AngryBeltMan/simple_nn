use crate::nn::NeuralNetwork;
// #[test]
// fn nn_creation_test() { let nn = NeuralNetwork::new_random([22,10,5]);
//     eprintln!("{nn:#?}");
//     assert_eq!(nn.layers[0].len(), 10);
//     for layer in &nn.layers[0] {
//         // each node should each have ten connections
//         assert_eq!(layer.len(), 22);
//     }
//     assert_eq!(nn.layers[1].len(), 5);
//     for layer in &nn.layers[1] {
//         assert_eq!(layer.len(), 10);
//     }
//     assert_eq!(nn.biases.len(), 2);
//     assert_eq!(nn.biases[0].len(), 10);
//     assert_eq!(nn.biases[1].len(), 5);
// }
// #[test]
// fn basic_nn() {
//     let mut nn = NeuralNetwork::new_random([3,2,1]);
//     nn.evaluate(&vec![0.5,0.6,0.7]);
//     eprintln!("{nn:#?}");
//     panic!()
// }
// #[test]
// fn nn_eval_test() {
//     let mut nn = NeuralNetwork::new_random([22,10,5]);
//     let res = nn.evaluate(&vec![0.6;22]);
//     eprintln!("{res:#?}");
// }
//
#[test]
fn nn_train() {
    let mut nn = NeuralNetwork::new_random([5,4,3, 2]);
    let data = vec![ vec![1.,2.,3.,4., 5.] ];
    let expected_output = vec![ vec![2., 1.] ];
    // eprintln!("{:#?}", nn);
    for _ in 0..100 {
        nn.train_nn(0.05, data.clone(), expected_output.clone());
    }
    // eprintln!("{:#?}", nn);
}
