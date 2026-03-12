use serde::{Deserialize, Serialize};

use crate::{HeError, HeResult};

/// Supported HE scheme families for the MVP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeScheme {
    /// Simulation of CKKS-style approximate arithmetic over vectors.
    CkksSimulation,
}

/// Homomorphic activation functions that can be approximated with
/// low-degree polynomials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    Identity,
    ReluApprox,
    SigmoidApprox,
    TanhApprox,
}

impl Activation {
    /// Polynomial coefficients in ascending order.
    pub fn polynomial(self) -> &'static [f64] {
        match self {
            Activation::Identity => &[0.0, 1.0],
            Activation::ReluApprox => &[0.0, 0.5, 0.125],
            Activation::SigmoidApprox => &[0.5, 0.197, 0.0, -0.004],
            Activation::TanhApprox => &[0.0, 0.815, 0.0, -0.081],
        }
    }
}

/// Tunable simulation parameters for encrypted arithmetic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeParameters {
    pub scheme: HeScheme,
    pub slots: usize,
    pub scaling_factor: f64,
    pub max_noise: f64,
    pub multiplicative_depth: usize,
}

impl HeParameters {
    pub fn medical_imaging() -> Self {
        Self {
            scheme: HeScheme::CkksSimulation,
            slots: 16_384,
            scaling_factor: 1_024.0,
            max_noise: 1e-3,
            multiplicative_depth: 6,
        }
    }

    pub fn validate(&self) -> HeResult<()> {
        if self.slots == 0 {
            return Err(HeError::KeyGen("slots must be greater than zero".to_string()));
        }
        if self.scaling_factor <= 0.0 {
            return Err(HeError::KeyGen(
                "scaling_factor must be greater than zero".to_string(),
            ));
        }
        if self.max_noise < 0.0 {
            return Err(HeError::KeyGen(
                "max_noise must be non-negative".to_string(),
            ));
        }
        if self.multiplicative_depth == 0 {
            return Err(HeError::KeyGen(
                "multiplicative_depth must be at least one".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for HeParameters {
    fn default() -> Self {
        Self::medical_imaging()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters_are_valid() {
        HeParameters::default().validate().unwrap();
    }

    #[test]
    fn test_relu_approximation_is_monotonic_near_origin() {
        let coeffs = Activation::ReluApprox.polynomial();
        let eval = |x: f64| coeffs[0] + coeffs[1] * x + coeffs[2] * x * x;

        assert!(eval(1.0) > eval(0.0));
        assert!(eval(0.0) >= eval(-1.0));
    }
}
