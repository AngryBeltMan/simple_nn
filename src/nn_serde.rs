type DynError = Box<dyn std::error::Error>;
use std::{fs::File, io::{Write, Read}};
use serde_json::{from_str, to_string};

use crate::nn::NeuralNetwork;
/// Serde json tools
pub trait NNSerde {

    /// save the current nn so it can be used later
    /// # Example
    /// ```
    /// let nn = NeuralNetword::new_random([3,2,4]);
    /// nn.save("save.json");
    /// ```
    fn save(&self, path: &str) -> Result<(), DynError>;

    /// Read the neural network from a previous save
    /// # Example
    /// ```
    /// let nn = NeuralNetword::frem_previous_save("save.json");
    /// ```
    fn from_previous_save(path:&str) -> Result<NeuralNetwork, DynError>;
}

impl NNSerde for NeuralNetwork {
    fn save(&self, path: &str) -> Result<(), DynError> {
        let mut file = File::create(path)?;
        file.write(to_string(self)?.as_bytes())?;
        Ok(())
    }
    fn from_previous_save(path:&str) -> Result<NeuralNetwork, DynError> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        eprintln!("{:#?}", contents);
        let nn: Self = from_str(&contents)?;
        Ok(nn)
    }

}
