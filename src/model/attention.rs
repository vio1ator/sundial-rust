//! Self-attention layer for Sundial
//!
//! Implements multi-head attention with rotary positional embeddings.
//! Matches the SundialAttention from the Python implementation.

use crate::debug_utils;
use crate::model::rope::SundialRotaryEmbedding;
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

/// Attention configuration
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub attention_dropout: f64,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub layer_idx: Option<usize>,
}

/// Multi-head self-attention with RoPE
pub struct SundialAttention {
    layer_idx: Option<usize>,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    attention_dropout: f64,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: SundialRotaryEmbedding,
}

impl SundialAttention {
    /// Create a new attention layer
    pub fn new(config: &AttentionConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;

        // Q, K, V projections with bias=True
        let q_weight = vb.get((hidden_size, hidden_size), "q_proj.weight")?;
        let q_bias = vb.get(hidden_size, "q_proj.bias")?;
        let q_proj = Linear::new(q_weight, Some(q_bias));

        let k_weight = vb.get((hidden_size, hidden_size), "k_proj.weight")?;
        let k_bias = vb.get(hidden_size, "k_proj.bias")?;
        let k_proj = Linear::new(k_weight, Some(k_bias));

        let v_weight = vb.get((hidden_size, hidden_size), "v_proj.weight")?;
        let v_bias = vb.get(hidden_size, "v_proj.bias")?;
        let v_proj = Linear::new(v_weight, Some(v_bias));

        // O projection with bias=False
        let o_weight = vb.get((hidden_size, hidden_size), "o_proj.weight")?;
        let o_proj = Linear::new(o_weight, None);

        // Rotary embeddings
        let rotary_emb = SundialRotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            vb.device(),
        )?;

        Ok(Self {
            layer_idx: config.layer_idx,
            hidden_size,
            num_heads,
            head_dim,
            attention_dropout: config.attention_dropout,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
        })
    }

    /// Apply scaled dot-product attention
    fn scaled_dot_product_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        dropout_p: f64,
        is_training: bool,
    ) -> Result<Tensor> {
        // query, key, value: [bs, num_heads, seq_len, head_dim]

        // Debug: print query/key/value stats
        if std::env::var("SUNDIAL_DEBUG").is_ok() {
            debug_utils::debug_tensor("attention_query", query);
            debug_utils::debug_tensor("attention_key", key);
            debug_utils::debug_tensor("attention_value", value);
        }

        // Compute attention scores: Q @ K^T / sqrt(head_dim)
        let scale = (query.dim(3)? as f64).sqrt().recip() as f32;
        let attn_scores = query
            .matmul(&key.t()?)?
            .broadcast_mul(&Tensor::new(scale, query.device())?);

        // Apply causal mask if provided
        let attn_weights = match &attention_mask {
            Some(mask) => attn_scores?.add(mask)?,
            None => attn_scores?,
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&attn_weights, 3)?;

        // Apply dropout if in training mode
        let attn_weights = if is_training && dropout_p > 0.0 {
            // Simple dropout: multiply by (1 - dropout_p) and scale
            // In practice, you'd add random masking
            let scale = 1.0 / (1.0 - dropout_p) as f32;
            attn_weights.broadcast_mul(&Tensor::new(scale, attn_weights.device())?)?
        } else {
            attn_weights
        };

        // Apply attention to values
        let output = attn_weights.matmul(value)?;

        // Debug: print attention output
        if std::env::var("SUNDIAL_DEBUG").is_ok() {
            debug_utils::debug_tensor("attention_output", &output);
        }

        Ok(output)
    }

    /// Create causal attention mask
    pub fn create_causal_mask(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
        // Create a causal mask where future positions are masked out
        let mut mask = vec![vec![f32::NEG_INFINITY; seq_len]; seq_len];

        for i in 0..seq_len {
            for j in 0..=i {
                mask[i][j] = 0.0;
            }
        }

        Tensor::from_vec(
            mask.into_iter().flatten().collect(),
            (seq_len, seq_len),
            device,
        )
    }
}

impl Module for SundialAttention {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (bsz, q_len, _) = hidden_states.dims3()?;

        // Project to Q, K, V
        let query_states = self.q_proj.forward(hidden_states)?;
        let key_states = self.k_proj.forward(hidden_states)?;
        let value_states = self.v_proj.forward(hidden_states)?;

        // Reshape for multi-head attention: [bsz, num_heads, seq_len, head_dim]
        let query_states = query_states
            .reshape((bsz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key_states = key_states
            .reshape((bsz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let value_states = value_states
            .reshape((bsz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Debug: print before RoPE
        if std::env::var("SUNDIAL_DEBUG").is_ok() {
            debug_utils::debug_tensor("attention_q_before_rope", &query_states);
            debug_utils::debug_tensor("attention_k_before_rope", &key_states);
        }

        // Apply rotary embeddings
        let (query_states, key_states) = self.rotary_emb.forward(
            &query_states,
            &key_states,
            None, // Use default position IDs
        )?;

        // Debug: print after RoPE
        if std::env::var("SUNDIAL_DEBUG").is_ok() {
            debug_utils::debug_tensor("attention_q_after_rope", &query_states);
            debug_utils::debug_tensor("attention_k_after_rope", &key_states);
        }

        // Create causal mask
        let causal_mask = Self::create_causal_mask(q_len, hidden_states.device())?;
        // Reshape mask for broadcasting: [1, 1, seq_len, seq_len]
        // Candle doesn't support full broadcasting, so we need to expand manually
        let causal_mask = causal_mask.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq_len, seq_len]
                                                                   // Expand to [bsz, num_heads, seq_len, seq_len]
        let causal_mask = causal_mask.repeat(&[bsz, self.num_heads, 1, 1])?;

        // Apply attention
        let attn_output = Self::scaled_dot_product_attention(
            &query_states,
            &key_states,
            &value_states,
            Some(&causal_mask),
            self.attention_dropout,
            false, // is_training
        )?;

        // Reshape back: [bsz, seq_len, hidden_size]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((bsz, q_len, self.hidden_size))?;

        // Output projection
        self.o_proj.forward(&attn_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_attention_creation() {
        let config = AttentionConfig {
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            attention_dropout: 0.1,
            max_position_embeddings: 10000,
            rope_theta: 10000.0,
            layer_idx: Some(0),
        };

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);

        let attn = SundialAttention::new(&config, vb);
        assert!(attn.is_ok());
    }

    #[test]
    fn test_attention_forward() {
        let config = AttentionConfig {
            hidden_size: 64,
            num_heads: 4,
            head_dim: 16,
            attention_dropout: 0.0,
            max_position_embeddings: 1000,
            rope_theta: 10000.0,
            layer_idx: Some(0),
        };

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);
        let attn = SundialAttention::new(&config, vb).unwrap();

        // Input: [batch=2, seq_len=10, hidden=64]
        let input = Tensor::randn(0.0f32, 1.0f32, (2, 10, 64), &device).unwrap();
        let output = attn.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 64]);
    }

    #[test]
    fn test_causal_mask() {
        let device = Device::Cpu;
        let mask = SundialAttention::create_causal_mask(4, &device).unwrap();

        // Check that upper triangle is -inf and lower triangle (including diagonal) is 0
        let mask_vec: Vec<Vec<f32>> = mask.to_vec2().unwrap();

        assert_eq!(mask_vec[0][0], 0.0);
        assert_eq!(mask_vec[0][1], f32::NEG_INFINITY);
        assert_eq!(mask_vec[1][0], 0.0);
        assert_eq!(mask_vec[1][1], 0.0);
        assert_eq!(mask_vec[1][2], f32::NEG_INFINITY);
    }
}
