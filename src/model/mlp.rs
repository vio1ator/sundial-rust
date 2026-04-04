//! MLP layer for Sundial using SwiGLU activation
//!
//! Matches the SundialMLP from the Python implementation.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

/// SwiGLU-style MLP layer
pub struct SundialMLP {
    hidden_size: usize,
    intermediate_size: usize,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: String,
}

impl SundialMLP {
    /// Create a new MLP layer
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        hidden_act: &str,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        // gate_proj: hidden_size -> intermediate_size (bias=False)
        let gate_weight = vb.get((intermediate_size, hidden_size), "gate_proj.weight")?;
        let gate_proj = Linear::new(gate_weight, None);

        // up_proj: hidden_size -> intermediate_size (bias=False)
        let up_weight = vb.get((intermediate_size, hidden_size), "up_proj.weight")?;
        let up_proj = Linear::new(up_weight, None);

        // down_proj: intermediate_size -> hidden_size (bias=False)
        let down_weight = vb.get((hidden_size, intermediate_size), "down_proj.weight")?;
        let down_proj = Linear::new(down_weight, None);

        Ok(Self {
            hidden_size,
            intermediate_size,
            gate_proj,
            up_proj,
            down_proj,
            act_fn: hidden_act.to_string(),
        })
    }

    /// Apply SiLU activation
    fn silu(x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::sigmoid(x)?.mul(x)
    }

    /// Apply the activation function
    fn apply_activation(x: &Tensor, act_fn: &str) -> Result<Tensor> {
        match act_fn {
            "silu" => Self::silu(x),
            "relu" => x.relu(),
            "gelu" => x.gelu(),
            _ => Err(candle_core::Error::Msg(format!(
                "Unsupported activation: {}",
                act_fn
            ))),
        }
    }
}

impl Module for SundialMLP {
    /// Forward pass: down_proj(act(gate_proj(x)) * up_proj(x))
    fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        // gate_proj(x)
        let gate = self.gate_proj.forward(hidden_state)?;

        // up_proj(x)
        let up = self.up_proj.forward(hidden_state)?;

        // act(gate) * up
        let activated = Self::apply_activation(&gate, &self.act_fn)?;
        let gated = activated.mul(&up)?;

        // down_proj(gated)
        self.down_proj.forward(&gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_mlp_creation() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);

        let mlp = SundialMLP::new(768, 3072, "silu", vb);
        assert!(mlp.is_ok());
    }

    #[test]
    fn test_mlp_forward() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);
        let mlp = SundialMLP::new(64, 128, "silu", vb).unwrap();

        // Input: [batch=2, seq_len=10, hidden=64]
        let input = Tensor::randn(0.0f32, 1.0f32, (2, 10, 64), &device).unwrap();
        let output = mlp.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 64]);
    }

    #[test]
    fn test_silu_activation() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![-1.0f32, 0.0f32, 1.0f32, 2.0f32], (4,), &device).unwrap();

        let result = SundialMLP::silu(&x).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();

        // SiLU(x) = x * sigmoid(x)
        // SiLU(-1) ≈ -0.269, SiLU(0) = 0, SiLU(1) ≈ 0.731, SiLU(2) ≈ 1.762
        assert!((values[0] - (-0.2689)).abs() < 0.01);
        assert!((values[1] - 0.0).abs() < 0.01);
        assert!((values[2] - 0.7311).abs() < 0.01);
        assert!((values[3] - 1.7616).abs() < 0.01);
    }
}
