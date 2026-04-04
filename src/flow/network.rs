//! Flow matching network
//!
//! The main network for predicting velocity in flow matching.
//! Matches the SimpleMLPAdaLN from the Python implementation.

use crate::flow::resblock::ResBlock;
use crate::flow::timestep_embed::TimestepEmbedder;
use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module};

/// Simple MLP with Adaptive Layer Normalization for flow matching
pub struct SimpleMLPAdaLN {
    in_channels: usize,
    model_channels: usize,
    out_channels: usize,
    num_res_blocks: usize,

    time_embed: TimestepEmbedder,
    cond_embed: Linear,
    input_proj: Linear,
    res_blocks: Vec<ResBlock>,
    final_layer: FinalLayer,
}

/// Final layer with modulation
pub struct FinalLayer {
    norm_final: candle_nn::LayerNorm,
    linear: Linear,
    ada_ln_modulation: Linear,
}

impl SimpleMLPAdaLN {
    /// Create a new flow matching network
    pub fn new(
        in_channels: usize,
        model_channels: usize,
        out_channels: usize,
        z_channels: usize,
        num_res_blocks: usize,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        // Timestep embedding
        let time_embed = TimestepEmbedder::new(model_channels, 256, vb.pp("time_embed"))?;

        // Condition embedding: z_channels -> model_channels
        let cond_embed = Linear::new(
            vb.get((model_channels, z_channels), "cond_embed.weight")?,
            Some(vb.get(model_channels, "cond_embed.bias")?),
        );

        // Input projection: in_channels -> model_channels
        let input_proj = Linear::new(
            vb.get((model_channels, in_channels), "input_proj.weight")?,
            Some(vb.get(model_channels, "input_proj.bias")?),
        );

        // Residual blocks
        let mut res_blocks = Vec::with_capacity(num_res_blocks);
        for i in 0..num_res_blocks {
            let block = ResBlock::new(model_channels, vb.pp(format!("res_blocks.{}", i)))?;
            res_blocks.push(block);
        }

        // Final layer
        let final_layer = FinalLayer::new(model_channels, out_channels, vb.pp("final_layer"))?;

        Ok(Self {
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            time_embed,
            cond_embed,
            input_proj,
            res_blocks,
            final_layer,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, in_channels]
    /// * `t` - Timestep tensor of shape [batch]
    /// * `c` - Condition tensor of shape [batch, z_channels]
    ///
    /// # Returns
    /// Predicted velocity of shape [batch, out_channels]
    pub fn forward(&self, x: &Tensor, t: &Tensor, c: &Tensor) -> Result<Tensor> {
        // Project input
        let mut x = self.input_proj.forward(x)?;

        // Embed timestep
        let t = self.time_embed.forward(t)?;

        // Embed condition
        let c = self.cond_embed.forward(c)?;

        // Combine timestep and condition
        let y = t.add(&c)?;

        // Apply residual blocks
        for block in &self.res_blocks {
            x = block.forward_resblock(&x, &y)?;
        }

        // Apply final layer
        self.final_layer.forward(&x, &y)
    }
}

impl FinalLayer {
    fn new(model_channels: usize, out_channels: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        // Note: norm_final is not in the safetensors, so we create it with default initialization
        // LayerNorm with weight=ones and bias=zeros
        let device = vb.device();
        let norm_weight = Tensor::ones(model_channels, DType::F32, device)?;
        let norm_bias = Tensor::zeros(model_channels, DType::F32, device)?;
        let norm_final = candle_nn::LayerNorm::new(norm_weight, norm_bias, 1e-6);

        let linear = Linear::new(
            vb.get((out_channels, model_channels), "linear.weight")?,
            Some(vb.get(out_channels, "linear.bias")?),
        );

        let ada_ln_modulation = Linear::new(
            vb.get(
                (2 * model_channels, model_channels),
                "adaLN_modulation.1.weight",
            )?,
            Some(vb.get(2 * model_channels, "adaLN_modulation.1.bias")?),
        );

        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        // Get shift and scale from modulation
        // Note: adaLN_modulation is Sequential(SiLU, Linear) in Python
        let c_silu = c.silu()?;
        let modulated = self.ada_ln_modulation.forward(&c_silu)?;
        let chunks = modulated.chunk(2, 1)?;
        let shift = &chunks[0];
        let scale = &chunks[1];

        // Normalize and modulate
        let normalized = self.norm_final.forward(x)?;
        let modulated = normalized.mul(
            &scale
                .broadcast_add(&Tensor::ones_like(shift)?)?
                .broadcast_add(shift)?,
        )?;

        // Linear projection
        self.linear.forward(&modulated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_flow_network_creation() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);

        let net = SimpleMLPAdaLN::new(720, 768, 720, 768, 3, vb);
        assert!(net.is_ok());
    }

    #[test]
    fn test_flow_network_forward() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);
        let net = SimpleMLPAdaLN::new(64, 128, 64, 128, 2, vb).unwrap();

        // Input: [batch=4, in_channels=64]
        let x = Tensor::randn(0.0f32, 1.0f32, (4, 64), &device).unwrap();
        // Timestep: [batch=4]
        let t = Tensor::from_vec(vec![0.0, 0.25, 0.5, 1.0], (4,), &device).unwrap();
        // Condition: [batch=4, z_channels=128]
        let c = Tensor::randn(0.0f32, 1.0f32, (4, 128), &device).unwrap();

        let output = net.forward(&x, &t, &c).unwrap();

        assert_eq!(output.dims(), &[4, 64]);
    }
}
