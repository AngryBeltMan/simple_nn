use simple_nn::*;
use rand::Rng;
use rand::prelude::*;
use std::fs::File;
use std::io::*;
type ImageData = Vec<f64>;
type ImageAnswer = Vec<f64>;
fn str_to_num(number: &str) -> u8 {
    if number == "0" {
        return 0;
    } else if number == "255" {
        return 255;
    } else {
        number.parse::<u8>().unwrap_or(0)
    }
}
fn open_data_set() -> Result<(Vec<ImageData>, Vec<ImageAnswer>)> {
    let mut rng = thread_rng();
    let mut contents = String::new();
    let mut file = File::open("A_Z Handwritten Data.csv").expect("could not find training set you can download it here 'https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format/download?datasetVersionNumber=5'");
    file.read_to_string(&mut contents).unwrap();
    let mut answers = vec![];
    let mut data: Vec<&str> = contents.split('\n')
        .into_iter().collect();
    data.shuffle(&mut rng);
    let data: Vec<Vec<f64>> = data.into_iter()
    .map(|row| {
            let mut row = row.split(',');
            let a = row.next().unwrap();
            answers.push((0..26).into_iter().map(|i| (i == str_to_num(a) ) as u8 as f64).collect());
            row.map(|n| {
                str_to_num(n) as f64 / 255.
            }).collect()
        }).collect();
    Ok((data, answers))
}
fn main() {
    let data_set = open_data_set().unwrap();
    println!("{:#?}",&data_set.1[0..10]);
    let mut nn = nn::NeuralNetwork::new_random([784, 42, 36, 26]);
    // nn.test_nn::<Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>>(&data_set.0, data_set.1);

    nn.test_nn(&data_set.0, &data_set.1);
    for training_chunks in data_set.0.chunks(1500).zip(data_set.1.chunks(1500)) {
        nn.train_nn(0.025, training_chunks.0.to_vec(), training_chunks.1.to_vec());
    }
}
