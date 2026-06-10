use std::time::Instant;

use fedlearn_core::error::{FedError, FedResult};
use fedlearn_core::model::{
    LayerWeights, LocalTrainConfig, ModelUpdate, ModelWeights, UpdateMetadata,
};
use he_inference::{Activation, DenseLayer, EncryptedModel};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const EPSILON: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputKind {
    Binary,
    Multiclass,
}

#[derive(Debug, Clone)]
pub struct DenseClassifier {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub output_kind: OutputKind,
    hidden_weights: Vec<Vec<f32>>,
    hidden_bias: Vec<f32>,
    output_weights: Vec<Vec<f32>>,
    output_bias: Vec<f32>,
}

impl DenseClassifier {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        output_kind: OutputKind,
        seed: u64,
    ) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let hidden_scale = (2.0 / input_dim.max(1) as f32).sqrt();
        let output_scale = (2.0 / hidden_dim.max(1) as f32).sqrt();

        let hidden_weights = (0..hidden_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| rng.gen_range(-hidden_scale..hidden_scale))
                    .collect()
            })
            .collect();
        let output_weights = (0..output_dim)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| rng.gen_range(-output_scale..output_scale))
                    .collect()
            })
            .collect();

        Self {
            input_dim,
            hidden_dim,
            output_dim,
            output_kind,
            hidden_weights,
            hidden_bias: vec![0.0; hidden_dim],
            output_weights,
            output_bias: vec![0.0; output_dim],
        }
    }

    pub fn train_local(
        &mut self,
        client_id: &str,
        data: &[Vec<f32>],
        labels: &[Vec<f32>],
        config: &LocalTrainConfig,
    ) -> FedResult<ModelUpdate> {
        validate_dataset(
            data,
            labels,
            self.input_dim,
            self.output_dim,
            self.output_kind,
        )?;

        let started = Instant::now();
        let batch_size = config.batch_size.max(1);

        for _ in 0..config.epochs {
            for (batch_data, batch_labels) in data.chunks(batch_size).zip(labels.chunks(batch_size))
            {
                self.train_batch(batch_data, batch_labels, config.learning_rate as f32)?;
            }
        }

        let (loss, _) = self.evaluate(data, labels)?;

        Ok(ModelUpdate {
            client_id: client_id.to_string(),
            weights: self.get_weights(),
            num_samples: data.len(),
            loss,
            metadata: UpdateMetadata {
                local_epochs: config.epochs,
                learning_rate: config.learning_rate,
                batch_size: config.batch_size,
                round: config.round,
                training_time_ms: started.elapsed().as_millis() as u64,
            },
        })
    }

    pub fn evaluate(&self, data: &[Vec<f32>], labels: &[Vec<f32>]) -> FedResult<(f64, f64)> {
        validate_dataset(
            data,
            labels,
            self.input_dim,
            self.output_dim,
            self.output_kind,
        )?;

        let mut total_loss = 0.0f64;
        let mut correct = 0usize;

        for (sample, target) in data.iter().zip(labels.iter()) {
            let (_, _, output) = self.forward(sample)?;
            total_loss += sample_loss(&output, target, self.output_kind) as f64;

            let is_correct = match self.output_kind {
                OutputKind::Binary => {
                    let prediction = if output[0] >= 0.5 { 1.0 } else { 0.0 };
                    (prediction - target[0]).abs() < 0.5
                }
                OutputKind::Multiclass => argmax(&output) == argmax(target),
            };

            correct += usize::from(is_correct);
        }

        let count = data.len() as f64;
        Ok((total_loss / count, correct as f64 / count))
    }

    pub fn get_weights(&self) -> ModelWeights {
        let mut layers = Vec::with_capacity(4);
        layers.push(LayerWeights {
            name: "hidden.weight".to_string(),
            shape: vec![self.hidden_dim, self.input_dim],
            data: flatten_matrix(&self.hidden_weights),
        });
        layers.push(LayerWeights {
            name: "hidden.bias".to_string(),
            shape: vec![self.hidden_dim],
            data: self.hidden_bias.clone(),
        });
        layers.push(LayerWeights {
            name: "output.weight".to_string(),
            shape: vec![self.output_dim, self.hidden_dim],
            data: flatten_matrix(&self.output_weights),
        });
        layers.push(LayerWeights {
            name: "output.bias".to_string(),
            shape: vec![self.output_dim],
            data: self.output_bias.clone(),
        });

        ModelWeights {
            num_params: layers.iter().map(|layer| layer.data.len()).sum(),
            layers,
        }
    }

    pub fn set_weights(&mut self, weights: &ModelWeights) -> FedResult<()> {
        if weights.layers.len() != 4 {
            return Err(FedError::DimensionMismatch {
                expected: 4,
                got: weights.layers.len(),
            });
        }

        let hidden_weight = find_layer(weights, "hidden.weight")?;
        let hidden_bias = find_layer(weights, "hidden.bias")?;
        let output_weight = find_layer(weights, "output.weight")?;
        let output_bias = find_layer(weights, "output.bias")?;

        validate_layer(hidden_weight, &[self.hidden_dim, self.input_dim])?;
        validate_layer(hidden_bias, &[self.hidden_dim])?;
        validate_layer(output_weight, &[self.output_dim, self.hidden_dim])?;
        validate_layer(output_bias, &[self.output_dim])?;

        self.hidden_weights =
            unflatten_matrix(&hidden_weight.data, self.hidden_dim, self.input_dim);
        self.hidden_bias = hidden_bias.data.clone();
        self.output_weights =
            unflatten_matrix(&output_weight.data, self.output_dim, self.hidden_dim);
        self.output_bias = output_bias.data.clone();

        Ok(())
    }

    pub fn num_params(&self) -> usize {
        self.hidden_dim * self.input_dim
            + self.hidden_dim
            + self.output_dim * self.hidden_dim
            + self.output_dim
    }

    pub fn to_encrypted_model(&self, labels: Vec<String>) -> FedResult<EncryptedModel> {
        let output_activation = match self.output_kind {
            OutputKind::Binary => Activation::SigmoidApprox,
            OutputKind::Multiclass => Activation::Identity,
        };

        EncryptedModel::new(
            vec![
                DenseLayer::new(
                    to_f64_matrix(&self.hidden_weights),
                    self.hidden_bias.iter().map(|value| *value as f64).collect(),
                    Activation::ReluApprox,
                )
                .map_err(|err| FedError::Training(err.to_string()))?,
                DenseLayer::new(
                    to_f64_matrix(&self.output_weights),
                    self.output_bias.iter().map(|value| *value as f64).collect(),
                    output_activation,
                )
                .map_err(|err| FedError::Training(err.to_string()))?,
            ],
            labels,
        )
        .map_err(|err| FedError::Training(err.to_string()))
    }

    fn train_batch(
        &mut self,
        data: &[Vec<f32>],
        labels: &[Vec<f32>],
        learning_rate: f32,
    ) -> FedResult<()> {
        let mut hidden_weight_grad = vec![vec![0.0; self.input_dim]; self.hidden_dim];
        let mut hidden_bias_grad = vec![0.0; self.hidden_dim];
        let mut output_weight_grad = vec![vec![0.0; self.hidden_dim]; self.output_dim];
        let mut output_bias_grad = vec![0.0; self.output_dim];

        for (sample, target) in data.iter().zip(labels.iter()) {
            let (hidden_linear, hidden_activation, output) = self.forward(sample)?;
            let mut output_error = output.clone();
            for (error, target_value) in output_error.iter_mut().zip(target.iter()) {
                *error -= *target_value;
            }

            for out_idx in 0..self.output_dim {
                output_bias_grad[out_idx] += output_error[out_idx];
                for hidden_idx in 0..self.hidden_dim {
                    output_weight_grad[out_idx][hidden_idx] +=
                        output_error[out_idx] * hidden_activation[hidden_idx];
                }
            }

            let mut hidden_error = vec![0.0; self.hidden_dim];
            for hidden_idx in 0..self.hidden_dim {
                let upstream: f32 = self
                    .output_weights
                    .iter()
                    .zip(output_error.iter())
                    .map(|(weights, error)| weights[hidden_idx] * error)
                    .sum();
                hidden_error[hidden_idx] = upstream
                    * if hidden_linear[hidden_idx] > 0.0 {
                        1.0
                    } else {
                        0.0
                    };
            }

            for hidden_idx in 0..self.hidden_dim {
                hidden_bias_grad[hidden_idx] += hidden_error[hidden_idx];
                for input_idx in 0..self.input_dim {
                    hidden_weight_grad[hidden_idx][input_idx] +=
                        hidden_error[hidden_idx] * sample[input_idx];
                }
            }
        }

        let scale = learning_rate / data.len().max(1) as f32;
        for hidden_idx in 0..self.hidden_dim {
            self.hidden_bias[hidden_idx] -= scale * hidden_bias_grad[hidden_idx];
            for input_idx in 0..self.input_dim {
                self.hidden_weights[hidden_idx][input_idx] -=
                    scale * hidden_weight_grad[hidden_idx][input_idx];
            }
        }

        for out_idx in 0..self.output_dim {
            self.output_bias[out_idx] -= scale * output_bias_grad[out_idx];
            for hidden_idx in 0..self.hidden_dim {
                self.output_weights[out_idx][hidden_idx] -=
                    scale * output_weight_grad[out_idx][hidden_idx];
            }
        }

        Ok(())
    }

    fn forward(&self, input: &[f32]) -> FedResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if input.len() != self.input_dim {
            return Err(FedError::DimensionMismatch {
                expected: self.input_dim,
                got: input.len(),
            });
        }

        let hidden_linear: Vec<f32> = self
            .hidden_weights
            .iter()
            .zip(self.hidden_bias.iter())
            .map(|(row, bias)| dot(row, input) + *bias)
            .collect();
        let hidden_activation: Vec<f32> =
            hidden_linear.iter().map(|value| value.max(0.0)).collect();

        let output_linear: Vec<f32> = self
            .output_weights
            .iter()
            .zip(self.output_bias.iter())
            .map(|(row, bias)| dot(row, &hidden_activation) + *bias)
            .collect();

        let output = match self.output_kind {
            OutputKind::Binary => vec![sigmoid(output_linear[0])],
            OutputKind::Multiclass => softmax(&output_linear),
        };

        Ok((hidden_linear, hidden_activation, output))
    }
}

fn validate_dataset(
    data: &[Vec<f32>],
    labels: &[Vec<f32>],
    input_dim: usize,
    output_dim: usize,
    output_kind: OutputKind,
) -> FedResult<()> {
    if data.is_empty() {
        return Err(FedError::Training("dataset must not be empty".to_string()));
    }
    if data.len() != labels.len() {
        return Err(FedError::Training(format!(
            "dataset size {} does not match label size {}",
            data.len(),
            labels.len()
        )));
    }
    for sample in data {
        if sample.len() != input_dim {
            return Err(FedError::DimensionMismatch {
                expected: input_dim,
                got: sample.len(),
            });
        }
    }
    for label in labels {
        let expected = match output_kind {
            OutputKind::Binary => 1,
            OutputKind::Multiclass => output_dim,
        };
        if label.len() != expected {
            return Err(FedError::DimensionMismatch {
                expected,
                got: label.len(),
            });
        }
    }
    Ok(())
}

fn sample_loss(prediction: &[f32], target: &[f32], output_kind: OutputKind) -> f32 {
    match output_kind {
        OutputKind::Binary => {
            let y = target[0].clamp(0.0, 1.0);
            let p = prediction[0].clamp(EPSILON, 1.0 - EPSILON);
            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        }
        OutputKind::Multiclass => prediction
            .iter()
            .zip(target.iter())
            .map(|(prob, y)| -y * prob.clamp(EPSILON, 1.0).ln())
            .sum(),
    }
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = values.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = exp_values.iter().sum();
    exp_values
        .iter()
        .map(|value| value / sum.max(EPSILON))
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn flatten_matrix(matrix: &[Vec<f32>]) -> Vec<f32> {
    matrix.iter().flat_map(|row| row.iter().copied()).collect()
}

fn unflatten_matrix(flat: &[f32], rows: usize, cols: usize) -> Vec<Vec<f32>> {
    flat.chunks(cols)
        .take(rows)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn validate_layer(layer: &LayerWeights, expected_shape: &[usize]) -> FedResult<()> {
    let expected_len: usize = expected_shape.iter().product();
    if layer.shape != expected_shape || layer.data.len() != expected_len {
        return Err(FedError::DimensionMismatch {
            expected: expected_len,
            got: layer.data.len(),
        });
    }
    Ok(())
}

fn find_layer<'a>(weights: &'a ModelWeights, name: &str) -> FedResult<&'a LayerWeights> {
    weights
        .layers
        .iter()
        .find(|layer| layer.name == name)
        .ok_or_else(|| FedError::Training(format!("missing layer {name}")))
}

fn to_f64_matrix(matrix: &[Vec<f32>]) -> Vec<Vec<f64>> {
    matrix
        .iter()
        .map(|row| row.iter().map(|value| *value as f64).collect())
        .collect()
}
