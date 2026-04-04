//! Residual block with adaptive layer normalization
//!
//! Matches the ResBlock from the Python implementation.

use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module};

/// Residual block with AdaLN modulation
pub struct ResBlock {
    channels: usize,
    in_ln: LayerNorm,
    mlp: (Linear, Linear),     // Two linear layers with SiLU
    ada_ln_modulation: Linear, // Single linear that outputs 3*channels for modulation
}

impl ResBlock {
    /// Create a new residual block
    pub fn new(channels: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        // Layer norm
        let in_ln = candle_nn::layer_norm(channels, 1e-6, vb.pp("in_ln"))?;

        // MLP: channels -> channels -> channels with SiLU
        let w1 = vb.get((channels, channels), "mlp.0.weight")?;
        let b1 = vb.get(channels, "mlp.0.bias")?;
        let linear1 = Linear::new(w1, Some(b1));

        let w2 = vb.get((channels, channels), "mlp.2.weight")?;
        let b2 = vb.get(channels, "mlp.2.bias")?;
        let linear2 = Linear::new(w2, Some(b2));

        // AdaLN modulation: channels -> 3*channels
        let ada_ln_modulation = Linear::new(
            vb.get((3 * channels, channels), "adaLN_modulation.1.weight")?,
            Some(vb.get(3 * channels, "adaLN_modulation.1.bias")?),
        );

        Ok(Self {
            channels,
            in_ln,
            mlp: (linear1, linear2),
            ada_ln_modulation,
        })
    }

    /// Create with proper Linear for modulation
    pub fn new_with_linear(channels: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        let in_ln = candle_nn::layer_norm(channels, 1e-6, vb.pp("in_ln"))?;

        let w1 = vb.get((channels, channels), "mlp.0.weight")?;
        let b1 = vb.get(channels, "mlp.0.bias")?;
        let linear1 = Linear::new(w1, Some(b1));

        let w2 = vb.get((channels, channels), "mlp.2.weight")?;
        let b2 = vb.get(channels, "mlp.2.bias")?;
        let linear2 = Linear::new(w2, Some(b2));

        let ada_ln_modulation = Linear::new(
            vb.get((3 * channels, channels), "adaLN_modulation.1.weight")?,
            Some(vb.get(3 * channels, "adaLN_modulation.1.bias")?),
        );

        Ok(Self {
            channels,
            in_ln,
            mlp: (linear1, linear2),
            ada_ln_modulation,
        })
    }

    /// Apply modulation to normalized input
    fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
        x.mul(
            &scale
                .broadcast_add(&Tensor::ones_like(shift)?)?
                .broadcast_add(shift)?,
        )
    }

    /// Forward pass with modulation from condition
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, channels]
    /// * `y` - Condition tensor of shape [batch, channels] for modulation
    pub(crate) fn forward_resblock(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        // Get modulation parameters: shift_mlp, scale_mlp, gate_mlp
        // Note: adaLN_modulation is Sequential(SiLU, Linear) in Python
        let y_silu = y.silu()?;
        let modulated = self.ada_ln_modulation.forward(&y_silu)?;
        let chunks = modulated.chunk(3, 1)?;
        let shift_mlp = &chunks[0];
        let scale_mlp = &chunks[1];
        let gate_mlp = &chunks[2];

        // Apply modulation to normalized input
        let normalized = self.in_ln.forward(x)?;
        let h = Self::modulate(&normalized, shift_mlp, scale_mlp)?;

        // Apply MLP
        let h = self.mlp.0.forward(&h)?;
        let h = h.silu()?;
        let h = self.mlp.1.forward(&h)?;

        // Apply gate and residual
        x.add(&h.mul(gate_mlp)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_resblock_creation() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);

        let block = ResBlock::new_with_linear(64, vb);
        assert!(block.is_ok());
    }

    #[test]
    fn test_resblock_forward() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);
        let block = ResBlock::new_with_linear(64, vb).unwrap();

        // Input and condition: [batch=4, channels=64]
        let x = Tensor::randn(0.0f32, 1.0f32, (4, 64), &device).unwrap();
        let y = Tensor::randn(0.0f32, 1.0f32, (4, 64), &device).unwrap();

        let output = block.forward_resblock(&x, &y).unwrap();

        assert_eq!(output.dims(), &[4, 64]);
    }
}
