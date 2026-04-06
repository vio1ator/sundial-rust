//! Decoder layer for Sundial
//!
//! Combines self-attention and MLP with residual connections and layer normalization.
//! Matches the SundialDecoderLayer from the Python implementation.

use crate::debug_utils;
use crate::model::attention::{AttentionConfig, SundialAttention};
use crate::model::mlp::SundialMLP;
use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Module};

/// Configuration for a decoder layer
#[derive(Debug, Clone)]
pub struct DecoderLayerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub attention_dropout: f64,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub layer_idx: usize,
}

/// A single decoder layer with self-attention and MLP
pub struct SundialDecoderLayer {
    self_attn: SundialAttention,
    ffn: SundialMLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl SundialDecoderLayer {
    /// Create a new decoder layer
    pub fn new(config: &DecoderLayerConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
        // Self-attention
        let attn_config = AttentionConfig {
            hidden_size: config.hidden_size,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim,
            attention_dropout: config.attention_dropout,
            max_position_embeddings: config.max_position_embeddings,
            rope_theta: config.rope_theta,
            layer_idx: Some(config.layer_idx),
        };
        let self_attn = SundialAttention::new(&attn_config, vb.pp("self_attn"))?;

        // MLP
        let ffn = SundialMLP::new(
            config.hidden_size,
            config.intermediate_size,
            &config.hidden_act,
            vb.pp("ffn_layer"),
        )?;

        // Layer norms
        let norm1 = candle_nn::layer_norm(config.hidden_size, 1e-5, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(config.hidden_size, 1e-5, vb.pp("norm2"))?;

        Ok(Self {
            self_attn,
            ffn,
            norm1,
            norm2,
        })
    }

    /// Forward pass with residual connections
    ///
    /// The layer applies:
    /// 1. norm1 -> self-attention -> residual
    /// 2. norm2 -> MLP -> residual
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let debug_mode = std::env::var("SUNDIAL_DEBUG").is_ok();
        let layer_idx = self.self_attn.get_layer_idx().unwrap_or(0);

        // Debug: print input
        if debug_mode {
            debug_utils::debug_tensor("decoder_layer_input", hidden_states);
        }

        let residual = hidden_states.clone();

        // Pre-normalization
        let normalized = self.norm1.forward(hidden_states)?;

        // Debug: print norm1 output
        if debug_mode {
            debug_utils::debug_tensor("decoder_layer_norm1", &normalized);
        }

        // Self-attention
        let attn_output = self.self_attn.forward(&normalized)?;

        // Debug: print and save attention output (matching Python naming)
        if debug_mode {
            let attn_name = format!("layer_{}_attention", layer_idx);
            debug_utils::debug_tensor(&attn_name, &attn_output);
            let _ = debug_utils::save_tensor_to_bin(&attn_name, &attn_output);
        }

        // Residual connection
        let hidden_states = residual.add(&attn_output)?;

        // Debug: print after first residual
        if debug_mode {
            debug_utils::debug_tensor("decoder_layer_after_attn_residual", &hidden_states);
        }

        // Second residual block
        let residual = hidden_states.clone();
        let normalized = self.norm2.forward(&hidden_states)?;

        // Debug: print norm2 output
        if debug_mode {
            debug_utils::debug_tensor("decoder_layer_norm2", &normalized);
        }

        let ffn_output = self.ffn.forward(&normalized)?;

        // Debug: print MLP output
        if debug_mode {
            debug_utils::debug_tensor("decoder_layer_mlp", &ffn_output);
        }

        let hidden_states = residual.add(&ffn_output)?;

        // Debug: print and save output (matching Python naming)
        if debug_mode {
            let output_name = format!("layer_{}_output", layer_idx);
            debug_utils::debug_tensor(&output_name, &hidden_states);
            let _ = debug_utils::save_tensor_to_bin(&output_name, &hidden_states);
        }

        Ok(hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_decoder_layer_creation() {
        let config = DecoderLayerConfig {
            hidden_size: 768,
            intermediate_size: 3072,
            num_attention_heads: 12,
            head_dim: 64,
            hidden_act: "silu".to_string(),
            attention_dropout: 0.1,
            max_position_embeddings: 10000,
            rope_theta: 10000.0,
            layer_idx: 0,
        };

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);

        let layer = SundialDecoderLayer::new(&config, vb);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_decoder_layer_forward() {
        let config = DecoderLayerConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            attention_dropout: 0.0,
            max_position_embeddings: 1000,
            rope_theta: 10000.0,
            layer_idx: 0,
        };

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);
        let layer = SundialDecoderLayer::new(&config, vb).unwrap();

        // Input: [batch=2, seq_len=10, hidden=64]
        let input = Tensor::randn(0.0f32, 1.0f32, (2, 10, 64), &device).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 64]);
    }
}
