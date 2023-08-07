use crate::{math::*, nn_activation::ActivationFunc};

type Weights = Vec<f64>;
type Layers = Vec<Vec<Weights>>;
type NodeOutputs = Vec<Vec<f64>>;
type Biases =  Vec<Vec<f64>>;

/// A simple neural network.
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Vec<Vec<Weights>>,
    node_outputs: NodeOutputs,
    biases: Biases,
    pub activation_function: ActivationFunc,
    layer_count: usize
}
impl NeuralNetwork {
    /// Creates a new empty neural network.
    /// The size argument determines how many layers the nueral network should have.
    pub fn new_empty(size: usize) -> Self {
        let layers: Layers = (1..size).into_iter().map(|_| vec![]).collect();
        let node_outputs: NodeOutputs = (0..size).into_iter().map(|_| vec![]).collect();
        let biases:Vec<Vec<f64>> = (1..size).into_iter().map(|_| vec![]).collect();
        Self { layers, biases, node_outputs, activation_function: ActivationFunc::Sigmoid, layer_count: size }
    }
    /// Returns a reference to the layer data.
    pub fn layers_data<'a>(&'a self) -> &'a Layers { &self.layers }

    /// Returns a reference to the node output data.
    pub fn node_output_data<'a>(&'a self) -> &'a NodeOutputs { &self.node_outputs }

    /// Returns a reference to the biases data.
    pub fn biases_data<'a>(&'a self) -> &'a Biases { &self.biases }

    /// Clears all of the node outputs.
    pub fn clear_outputs(&mut self) {
        for layer in 0..self.layer_count {
            self.node_outputs[layer] = vec![];
        }
    }

    /// Creates a new nueral network with random weights between -0.5 and 0.5.
    /// The length of the layers argument determines how many layers the neural network should
    /// have, and the numbers in each array determines how many nodes/neurons are in each layer.
    pub fn new_random<const LEN: usize>(layers: [usize; LEN]) -> Self {
        // makes sure there is an input and an output layer
        assert!(LEN > 1);
        let mut nn = Self::new_empty(LEN);
        for layer in 1..LEN {
            for _ in 0..layers[layer] {
                // sets all the inital biases to 1
                nn.biases[layer - 1].push(1.);
                nn.layers[layer - 1].push((0..layers[layer - 1]).into_iter().map(|_| {
                    (rand::random::<f64>() % 1.) - 0.5
                }).collect());
            }
        }
        nn
    }
    /// Change the nueral networks activation function.
    pub fn switch_activation(&mut self, activation_overwrite: ActivationFunc) {
        self.activation_function = activation_overwrite;
    }

    /// Returns the nueral networks prediction given an input.
    /// Will panic if the input does not match the length of the nn input layer.
    pub fn evaluate<'a>(&'a mut self, input: &Vec<f64>) -> &'a Vec<f64> {
        assert_eq!(self.layers[0][0].len(), input.len());
        // the output will be reused as input
        self.node_outputs[0] = input.to_vec();
        for layer in 0..self.layer_count - 1 {
            let mut nueron = 0;
            self.node_outputs[layer + 1] = self.layers[layer].iter().map(|lay| {
                let input = lay.iter().zip(&self.node_outputs[layer]).map(|x| {
                        (*x.0) * (*x.1)
                    }).sum::<f64>() + self.biases[layer][nueron];
                nueron += 1;
                match self.activation_function {
                    ActivationFunc::ReLU => reLU(input),
                    ActivationFunc::Sigmoid => sigmoid(input)
                }
            }).collect();
            // self.node_outputs[layer + 1] = next_output;
        }
        &self.node_outputs[self.layer_count - 1]
    }
    /// Compute the accuracy of the nn given the output and target value of the nn.
    pub fn compute_cost(output: Vec<f64>, target: Vec<f64>) -> f64 {
        output.iter().zip(&target).map(|f| {
            (f.1 - f.0).powf(2.)
        }).sum::<f64>() / target.len() as f64
    }
    // returns a list of indexs of items the nn correctly predicted
    pub fn test_nn(&mut self, data: Vec<Vec<f64>>, target: Vec<Vec<f64>>) -> Vec<usize> {
        let mut index = 0;
        data.iter().zip(&target)
            .filter_map(|img| {
                let prediction = self.evaluate(img.0);
                let correct = if prediction.maximum() == img.1.maximum() {
                    Some(index)
                } else {
                    None
                };
                index += 1;
                correct
            }).collect()
    }

    /// Update the inital weights given the delta value.
    #[inline]
    fn update_inital_weight(&mut self, delta_o: &Vec<f64>, learn_rate: f64) {
        let mut weight_change = element_wise_multiplication(&delta_o, &self.node_outputs[self.layer_count - 2]);
        scalar_multiplication(&mut weight_change, -learn_rate);
        matrix_addition(&mut self.layers[self.layer_count - 2], &weight_change);
    }
    /// Update the biases given the delta value.
    #[inline]
    fn update_biases(&mut self, delta_v: &Vec<f64>, learn_rate: f64) {
        let bias_change = delta_v.iter().map(|v| v * (-learn_rate)).collect::<Vec<f64>>();
        let _ = (&mut self.biases[self.layer_count - 2])
            .iter_mut()
            .zip(&bias_change)
            .map(|b| *b.0 += *b.1).collect::<()>();
    }
    /// Train the neural network.
    /// The learn rate determines how fast the nn should learn. The data argument is the list of
    /// input data and the target agument is the expected output for each given data.
    pub fn train_nn(&mut self, learn_rate: f64, data: Vec<Vec<f64>>, target: Vec<Vec<f64>>) {
        let mut correct = 0;
        let index = data.len();

        for training_image in data.into_iter().zip(target) {
            let mut weight_change;
            // gets nn prediction given the input
            let output = self.evaluate(&training_image.0);
            // adds one if the nn prediction is correct and 0 if it was wrong
            correct += (output.maximum() == training_image.1.maximum()) as usize;
            // the difference between the nn output and the expected value
            let delta_o = output.subtract(&training_image.1);
            self.update_inital_weight(&delta_o, learn_rate);
            self.update_biases(&delta_o, learn_rate);
            let mut delta_h = delta_o;
            // Begin the walk backwards across the nn updating the weights and biases
            for layer in 2..self.layer_count {
                // walks across the nueral network backwards
                let layer = self.layer_count - layer;
                // calculates the derivative of the activation function
                let h = &self.node_outputs[layer];
                let mut deriviative = h.iter().map(|v| {1. - v}).collect::<Vec<f64>>();
                vector_mult(&mut deriviative, h);

                // assign the delta_h value
                let transposed_weight_change = transposition(&self.layers[layer]);
                let res = matrix_mult(&transposed_weight_change, &vec_to_matrix(&delta_h));
                delta_h = matrix_to_vec(&res);
                vector_mult(&mut delta_h, &deriviative);

                // calculates how much the weights should change given the delta_h
                weight_change = element_wise_multiplication(&delta_h, &self.node_outputs[layer - 1]);
                scalar_multiplication(&mut weight_change, -learn_rate);

                // applies the change to the weights
                matrix_addition(&mut self.layers[layer - 1], &weight_change);

                self.update_biases(&delta_h, learn_rate);
                }
            }
            // calculates the accuracy of the nn
            println!("accuracy: {}", (correct as f32/ index as f32));
        }
    }
