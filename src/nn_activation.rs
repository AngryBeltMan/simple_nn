use serde_derive::*;
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ActivationFunc {
    ReLU,
    Sigmoid
}
