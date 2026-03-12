use he_core::{
    Activation,
    CiphertextVector,
    HeError,
    HeResult,
    ServerKey,
    apply_activation,
    linear_layer,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub activation: Activation,
}

impl DenseLayer {
    pub fn new(weights: Vec<Vec<f64>>, bias: Vec<f64>, activation: Activation) -> HeResult<Self> {
        if weights.is_empty() {
            return Err(HeError::Operation(
                "dense layer requires at least one output neuron".to_string(),
            ));
        }
        let input_dim = weights[0].len();
        if input_dim == 0 {
            return Err(HeError::Operation(
                "dense layer requires at least one input feature".to_string(),
            ));
        }
        if weights.iter().any(|row| row.len() != input_dim) {
            return Err(HeError::Operation(
                "dense layer weights must form a rectangular matrix".to_string(),
            ));
        }
        if weights.len() != bias.len() {
            return Err(HeError::Operation(format!(
                "bias length {} does not match output width {}",
                bias.len(),
                weights.len()
            )));
        }

        Ok(Self {
            weights,
            bias,
            activation,
        })
    }

    pub fn input_dim(&self) -> usize {
        self.weights[0].len()
    }

    pub fn output_dim(&self) -> usize {
        self.weights.len()
    }

    pub fn forward_plaintext(&self, input: &[f64]) -> HeResult<Vec<f64>> {
        if input.len() != self.input_dim() {
            return Err(HeError::Operation(format!(
                "input length {} does not match layer width {}",
                input.len(),
                self.input_dim()
            )));
        }

        let outputs: Vec<f64> = self
            .weights
            .iter()
            .zip(self.bias.iter())
            .map(|(row, bias)| {
                let affine = row
                    .iter()
                    .zip(input.iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f64>()
                    + *bias;
                apply_polynomial_scalar(affine, self.activation.polynomial())
            })
            .collect();

        Ok(outputs)
    }

    pub fn forward_encrypted(
        &self,
        input: &CiphertextVector,
        server_key: &ServerKey,
    ) -> HeResult<CiphertextVector> {
        let affine = linear_layer(input, &self.weights, &self.bias, server_key)?;
        apply_activation(&affine, self.activation, server_key)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedModel {
    pub layers: Vec<DenseLayer>,
    pub labels: Vec<String>,
    input_dim: usize,
}

impl EncryptedModel {
    pub fn new(layers: Vec<DenseLayer>, labels: Vec<String>) -> HeResult<Self> {
        if layers.is_empty() {
            return Err(HeError::Operation(
                "encrypted model requires at least one layer".to_string(),
            ));
        }

        for window in layers.windows(2) {
            let current = &window[0];
            let next = &window[1];
            if current.output_dim() != next.input_dim() {
                return Err(HeError::Operation(format!(
                    "layer output dimension {} does not match next input dimension {}",
                    current.output_dim(),
                    next.input_dim()
                )));
            }
        }

        let output_dim = layers.last().map(|layer| layer.output_dim()).unwrap_or_default();
        if !labels.is_empty() && labels.len() != output_dim {
            return Err(HeError::Operation(format!(
                "label count {} does not match output dimension {}",
                labels.len(),
                output_dim
            )));
        }

        Ok(Self {
            input_dim: layers[0].input_dim(),
            layers,
            labels,
        })
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.layers.last().map(|layer| layer.output_dim()).unwrap_or(0)
    }

    pub fn infer_plaintext(&self, input: &[f64]) -> HeResult<Vec<f64>> {
        if input.len() != self.input_dim {
            return Err(HeError::Operation(format!(
                "input length {} does not match model width {}",
                input.len(),
                self.input_dim
            )));
        }

        let mut activations = input.to_vec();
        for layer in &self.layers {
            activations = layer.forward_plaintext(&activations)?;
        }
        Ok(activations)
    }

    pub fn infer_encrypted(
        &self,
        input: &CiphertextVector,
        server_key: &ServerKey,
    ) -> HeResult<CiphertextVector> {
        if input.len() != self.input_dim {
            return Err(HeError::Operation(format!(
                "ciphertext length {} does not match model width {}",
                input.len(),
                self.input_dim
            )));
        }

        let mut activations = input.clone();
        for layer in &self.layers {
            activations = layer.forward_encrypted(&activations, server_key)?;
        }
        Ok(activations)
    }
}

fn apply_polynomial_scalar(value: f64, coefficients: &[f64]) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |acc, coeff| acc * value + coeff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use he_core::{HeParameters, decrypt_vector, encrypt_vector, generate_keys};

    #[test]
    fn test_encrypted_model_matches_plaintext_forward_pass() {
        let model = EncryptedModel::new(
            vec![
                DenseLayer::new(
                    vec![vec![0.5, -0.25], vec![1.0, 0.5]],
                    vec![0.1, -0.2],
                    Activation::ReluApprox,
                )
                .unwrap(),
                DenseLayer::new(
                    vec![vec![1.2, -0.7]],
                    vec![0.05],
                    Activation::SigmoidApprox,
                )
                .unwrap(),
            ],
            vec!["abnormal".to_string()],
        )
        .unwrap();

        let input = vec![0.8, -0.4];
        let keys = generate_keys(HeParameters::default()).unwrap();
        let ciphertext = encrypt_vector(&keys.public_key, &input).unwrap();

        let encrypted_output = model.infer_encrypted(&ciphertext, &keys.server_key).unwrap();
        let decrypted_output = decrypt_vector(&keys.client_key, &encrypted_output).unwrap();
        let plaintext_output = model.infer_plaintext(&input).unwrap();

        assert_eq!(decrypted_output.len(), 1);
        assert!((decrypted_output[0] - plaintext_output[0]).abs() < 1e-9);
    }

    #[test]
    fn test_model_validation_rejects_dimension_mismatch() {
        let err = EncryptedModel::new(
            vec![
                DenseLayer::new(vec![vec![1.0, 2.0]], vec![0.0], Activation::Identity).unwrap(),
                DenseLayer::new(vec![vec![1.0, 2.0, 3.0]], vec![0.0], Activation::Identity)
                    .unwrap(),
            ],
            vec![],
        )
        .unwrap_err();

        assert!(matches!(err, HeError::Operation(_)));
    }
}
