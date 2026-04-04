//! Transformer backbone for Sundial
//!
//! Stacks decoder layers and applies final normalization.
//! Matches the SundialModel from the Python implementation.

use crate::debug_utils;
use crate::model::config::SundialConfig;
use crate::model::decoder_layer::{DecoderLayerConfig, SundialDecoderLayer};
use crate::model::patch_embed::SundialPatchEmbedding;
use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Module};

/// Transformer backbone with patch embedding
pub struct SundialTransformer {
    embed_layer: SundialPatchEmbedding,
    layers: Vec<SundialDecoderLayer>,
    norm: LayerNorm,
    config: SundialConfig,
}

impl SundialTransformer {
    /// Create a new transformer backbone
    pub fn new(config: &SundialConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
        // Patch embedding layer
        let embed_layer = SundialPatchEmbedding::new(config, vb.pp("embed_layer"))?;

        // Decoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let layer_config = DecoderLayerConfig {
                hidden_size: config.hidden_size,
                intermediate_size: config.intermediate_size,
                num_attention_heads: config.num_attention_heads,
                head_dim: config.head_dim(),
                hidden_act: config.hidden_act.clone(),
                attention_dropout: config.dropout_rate,
                max_position_embeddings: config.max_position_embeddings,
                rope_theta: config.rope_theta,
                layer_idx,
            };
            layers.push(SundialDecoderLayer::new(
                &layer_config,
                vb.pp(format!("layers.{}", layer_idx)),
            )?);
        }

        // Final layer norm
        let norm = candle_nn::layer_norm(config.hidden_size, 1e-5, vb.pp("norm"))?;

        Ok(Self {
            embed_layer,
            layers,
            norm,
            config: config.clone(),
        })
    }

    /// Forward pass through the transformer
    ///
    /// # Arguments
    /// * `input_ids` - Input tensor of shape [batch_size, seq_len]
    ///
    /// # Returns
    /// Hidden states of shape [batch_size, num_patches, hidden_size]
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Debug: print input
        if std::env::var("SUNDIAL_DEBUG").is_ok() {
            debug_utils::debug_tensor("transformer_input", input_ids);
        }

        // Patch embedding
        let mut hidden_states = self.embed_layer.forward(input_ids)?;

        // Debug: print after patch embed
        if std::env::var("SUNDIAL_DEBUG").is_ok() {
            debug_utils::debug_tensor("transformer_after_patch_embed", &hidden_states);
        }

        // Pass through decoder layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states)?;

            // Debug: print after each layer
            if std::env::var("SUNDIAL_DEBUG").is_ok() {
                debug_utils::debug_tensor(
                    &format!("transformer_after_layer_{}", layer_idx),
                    &hidden_states,
                );
            }

            // Stop after specified layer for debugging
            if let Some(stop_layer) = std::env::var("SUNDIAL_DEBUG_LAYER")
                .ok()
                .and_then(|s| s.parse().ok())
            {
                if layer_idx >= stop_layer {
                    println!("[DEBUG] Stopping after layer {}", layer_idx);
                    break;
                }
            }
        }

        // Final normalization
        let output = self.norm.forward(&hidden_states)?;

        // Debug: print final output
        if std::env::var("SUNDIAL_DEBUG").is_ok() {
            debug_utils::debug_tensor("transformer_output", &output);
        }

        Ok(output)
    }

    /// Get the number of hidden layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the hidden size
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}

impl Module for SundialTransformer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_transformer_creation() {
        let config = SundialConfig {
            input_token_len: 16,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            ..Default::default()
        };

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);

        let transformer = SundialTransformer::new(&config, vb);
        assert!(transformer.is_ok());
    }

    #[test]
    fn test_transformer_forward() {
        let config = SundialConfig {
            input_token_len: 16,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            ..Default::default()
        };

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);
        let transformer = SundialTransformer::new(&config, vb).unwrap();

        // Input: [batch=2, seq_len=64] (4 patches of 16)
        let input = Tensor::randn(0.0f32, 1.0f32, (2, 64), &device).unwrap();
        let output = transformer.forward(&input).unwrap();

        // Output: [batch=2, num_patches=4, hidden=64]
        assert_eq!(output.dims(), &[2, 4, 64]);
    }

    #[test]
    fn test_full_length_input() {
        let config = SundialConfig {
            input_token_len: 16,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            ..Default::default()
        };

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);
        let transformer = SundialTransformer::new(&config, vb).unwrap();

        // Input: [batch=1, seq_len=2880] (180 patches)
        let input = Tensor::randn(0.0f32, 1.0f32, (1, 2880), &device).unwrap();
        let output = transformer.forward(&input).unwrap();

        // Output: [batch=1, num_patches=180, hidden=64]
        assert_eq!(output.dims(), &[1, 180, 64]);
    }
}
