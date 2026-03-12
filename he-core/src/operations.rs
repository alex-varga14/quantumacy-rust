use crate::encrypt::{CiphertextVector, ServerKey};
use crate::schemes::Activation;
use crate::{HeError, HeResult};

pub fn add_ciphertexts(left: &CiphertextVector, right: &CiphertextVector) -> HeResult<CiphertextVector> {
    ensure_compatible(left, right)?;

    Ok(CiphertextVector {
        key_id: left.key_id.clone(),
        encoded: left
            .encoded
            .iter()
            .zip(right.encoded.iter())
            .map(|(a, b)| a + b)
            .collect(),
        mask: left
            .mask
            .iter()
            .zip(right.mask.iter())
            .map(|(a, b)| a + b)
            .collect(),
        scale: left.scale.max(right.scale),
        noise_budget: left.noise_budget.min(right.noise_budget),
    })
}

pub fn add_plaintext(ciphertext: &CiphertextVector, plaintext: &[f64]) -> HeResult<CiphertextVector> {
    if ciphertext.len() != plaintext.len() {
        return Err(HeError::Operation(format!(
            "plaintext length {} does not match ciphertext length {}",
            plaintext.len(),
            ciphertext.len()
        )));
    }

    Ok(CiphertextVector {
        key_id: ciphertext.key_id.clone(),
        encoded: ciphertext
            .encoded
            .iter()
            .zip(plaintext.iter())
            .map(|(encoded, value)| encoded + value)
            .collect(),
        mask: ciphertext.mask.clone(),
        scale: ciphertext.scale,
        noise_budget: ciphertext.noise_budget,
    })
}

pub fn multiply_plaintext(ciphertext: &CiphertextVector, scalar: f64) -> HeResult<CiphertextVector> {
    Ok(CiphertextVector {
        key_id: ciphertext.key_id.clone(),
        encoded: ciphertext.encoded.iter().map(|value| value * scalar).collect(),
        mask: ciphertext.mask.iter().map(|value| value * scalar).collect(),
        scale: ciphertext.scale,
        noise_budget: ciphertext.noise_budget,
    })
}

pub fn multiply_ciphertexts(
    left: &CiphertextVector,
    right: &CiphertextVector,
    server_key: &ServerKey,
) -> HeResult<CiphertextVector> {
    ensure_compatible(left, right)?;
    let products: Vec<f64> = left
        .plaintext()
        .iter()
        .zip(right.plaintext().iter())
        .map(|(a, b)| a * b)
        .collect();

    server_key.refresh_ciphertext(
        &left.key_id,
        &products,
        left.scale.max(right.scale),
        1,
    )
}

pub fn apply_polynomial(
    ciphertext: &CiphertextVector,
    coefficients: &[f64],
    server_key: &ServerKey,
) -> HeResult<CiphertextVector> {
    if coefficients.is_empty() {
        return Err(HeError::Operation(
            "polynomial coefficients may not be empty".to_string(),
        ));
    }

    let values = ciphertext.plaintext();
    let transformed: Vec<f64> = values
        .iter()
        .map(|value| {
            coefficients
                .iter()
                .rev()
                .fold(0.0, |acc, coeff| acc * value + coeff)
        })
        .collect();

    server_key.refresh_ciphertext(&ciphertext.key_id, &transformed, ciphertext.scale, 1)
}

pub fn apply_activation(
    ciphertext: &CiphertextVector,
    activation: Activation,
    server_key: &ServerKey,
) -> HeResult<CiphertextVector> {
    apply_polynomial(ciphertext, activation.polynomial(), server_key)
}

pub fn linear_layer(
    input: &CiphertextVector,
    weights: &[Vec<f64>],
    bias: &[f64],
    server_key: &ServerKey,
) -> HeResult<CiphertextVector> {
    if weights.is_empty() {
        return Err(HeError::Operation("linear layer requires weights".to_string()));
    }
    if weights.len() != bias.len() {
        return Err(HeError::Operation(format!(
            "bias length {} does not match output width {}",
            bias.len(),
            weights.len()
        )));
    }
    if weights.iter().any(|row| row.len() != input.len()) {
        return Err(HeError::Operation(
            "weight row width does not match input size".to_string(),
        ));
    }

    let plaintext = input.plaintext();
    let outputs: Vec<f64> = weights
        .iter()
        .zip(bias.iter())
        .map(|(row, bias)| {
            row.iter()
                .zip(plaintext.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f64>()
                + *bias
        })
        .collect();

    server_key.refresh_ciphertext(&input.key_id, &outputs, input.scale, 1)
}

fn ensure_compatible(left: &CiphertextVector, right: &CiphertextVector) -> HeResult<()> {
    if left.key_id != right.key_id {
        return Err(HeError::Operation(format!(
            "ciphertext key ids differ ({} vs {})",
            left.key_id, right.key_id
        )));
    }
    if left.len() != right.len() {
        return Err(HeError::Operation(format!(
            "ciphertext lengths differ ({} vs {})",
            left.len(),
            right.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encrypt::{decrypt_vector, encrypt_vector, generate_keys};
    use crate::schemes::HeParameters;

    #[test]
    fn test_add_ciphertexts_roundtrip() {
        let keys = generate_keys(HeParameters::default()).unwrap();
        let left = encrypt_vector(&keys.public_key, &[1.0, 2.0]).unwrap();
        let right = encrypt_vector(&keys.public_key, &[0.5, -2.0]).unwrap();

        let sum = add_ciphertexts(&left, &right).unwrap();
        let decrypted = decrypt_vector(&keys.client_key, &sum).unwrap();

        assert_eq!(decrypted, vec![1.5, 0.0]);
    }

    #[test]
    fn test_linear_layer_matches_plaintext_math() {
        let keys = generate_keys(HeParameters::default()).unwrap();
        let input = encrypt_vector(&keys.public_key, &[2.0, -1.0]).unwrap();

        let output = linear_layer(
            &input,
            &[vec![1.0, 3.0], vec![0.5, -0.5]],
            &[0.25, 1.0],
            &keys.server_key,
        )
        .unwrap();

        let decrypted = decrypt_vector(&keys.client_key, &output).unwrap();
        assert!((decrypted[0] - (-0.75)).abs() < 1e-9);
        assert!((decrypted[1] - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_activation_polynomial_changes_values() {
        let keys = generate_keys(HeParameters::default()).unwrap();
        let input = encrypt_vector(&keys.public_key, &[-1.0, 0.0, 1.0]).unwrap();

        let activated = apply_activation(&input, Activation::SigmoidApprox, &keys.server_key).unwrap();
        let decrypted = decrypt_vector(&keys.client_key, &activated).unwrap();

        assert!(decrypted[0] < decrypted[2]);
        assert!((decrypted[1] - 0.5).abs() < 1e-9);
    }
}
