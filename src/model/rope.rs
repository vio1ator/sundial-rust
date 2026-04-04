//! Rotary Positional Embeddings (RoPE)
//!
//! Implements the rotary embedding mechanism used in Sundial for position-aware attention.
//! Matches the SundialRotaryEmbedding from the Python implementation.

use candle_core::{DType, Result, Tensor};
use std::sync::Arc;

/// Rotary positional embedding
pub struct SundialRotaryEmbedding {
    dim: usize,
    max_position_embeddings: usize,
    base: f64,
    inv_freq: Tensor,
    cos_cached: Arc<Tensor>,
    sin_cached: Arc<Tensor>,
}

impl SundialRotaryEmbedding {
    /// Create a new rotary embedding
    pub fn new(
        dim: usize,
        max_position_embeddings: usize,
        base: f64,
        device: &candle_core::Device,
    ) -> Result<Self> {
        // Calculate inverse frequencies: 1 / (base ^ (2i / dim))
        let mut inv_freq = Vec::with_capacity(dim / 2);
        for i in (0..dim).step_by(2) {
            let freq = base.powf(-(i as f64) / (dim as f64));
            inv_freq.push(1.0 / freq);
        }

        let inv_freq = Tensor::from_vec(inv_freq, (dim / 2,), device)?.to_dtype(DType::F32)?;

        // Pre-compute cos/sin caches
        let (cos_cached, sin_cached) =
            Self::_set_cos_sin_cache(&inv_freq, max_position_embeddings, device, DType::F32)?;

        Ok(Self {
            dim,
            max_position_embeddings,
            base,
            inv_freq,
            cos_cached: Arc::new(cos_cached),
            sin_cached: Arc::new(sin_cached),
        })
    }

    /// Set cos/sin cache for given sequence length
    fn _set_cos_sin_cache(
        inv_freq: &Tensor,
        seq_len: usize,
        device: &candle_core::Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        // Create position indices: [0, 1, 2, ..., seq_len-1]
        let t: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let t = Tensor::from_vec(t, (seq_len,), device)?;

        // freqs = t.outer(inv_freq)
        let freqs = t.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        // emb = cat([freqs, freqs], dim=-1)
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;

        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;

        Ok((cos, sin))
    }

    /// Rotate half of the tensor
    fn rotate_half(x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let last_dim = dims[dims.len() - 1];
        let half = last_dim / 2;
        let last_dim_idx = dims.len() - 1;

        let x1 = x.narrow(last_dim_idx, 0, half)?;
        let x2 = x.narrow(last_dim_idx, half, half)?;

        // Return: cat(-x2, x1)
        let neg_x2 = x2.neg()?;
        Tensor::cat(&[&neg_x2, &x1], last_dim_idx)?.contiguous()
    }

    /// Apply rotary positional embeddings to query and key
    ///
    /// # Arguments
    /// * `q` - Query tensor of shape [..., dim]
    /// * `k` - Key tensor of shape [..., dim]
    /// * `position_ids` - Position indices (optional, defaults to sequential)
    ///
    /// # Returns
    /// Tuple of (q_embed, k_embed)
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = match position_ids {
            Some(ids) => {
                // Use provided position IDs
                let max_pos: i64 = ids.max_keepdim(0)?.to_scalar()?;
                (max_pos.abs() + 1) as usize
            }
            None => {
                // Default to sequence length
                q.dims()[q.dims().len() - 2]
            }
        };

        // Get cached cos/sin for the required length
        let cos = self.cos_cached.narrow(0, 0, seq_len)?;
        let sin = self.sin_cached.narrow(0, 0, seq_len)?;

        // Reshape cos/sin to match q/k dimensions for broadcasting
        // q/k shape: [bs, num_heads, seq_len, head_dim]
        // cos/sin shape: [seq_len, 2*head_dim]
        // We need to reshape to [1, 1, seq_len, head_dim] for each half

        // Reshape cos/sin to [1, 1, seq_len, dim] and then expand to match query shape
        let cos = cos
            .reshape((1, 1, seq_len, self.dim))?
            .repeat(&[q.dim(0)?, q.dim(1)?, 1, 1])?;

        let sin = sin
            .reshape((1, 1, seq_len, self.dim))?
            .repeat(&[q.dim(0)?, q.dim(1)?, 1, 1])?;

        // Apply RoPE: q_embed = q * cos + rotate_half(q) * sin
        let q_embed = q.mul(&cos)?.add(&Self::rotate_half(q)?.mul(&sin)?)?;
        let k_embed = k.mul(&cos)?.add(&Self::rotate_half(k)?.mul(&sin)?)?;

        Ok((q_embed, k_embed))
    }

    /// Get the dimension of the embedding
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_rope_creation() {
        let device = Device::Cpu;
        let rope = SundialRotaryEmbedding::new(64, 1000, 10000.0, &device);
        assert!(rope.is_ok());
    }

    #[test]
    fn test_rope_forward() {
        let device = Device::Cpu;
        let rope = SundialRotaryEmbedding::new(64, 1000, 10000.0, &device).unwrap();

        // Create dummy query and key: [bs=2, heads=4, seq_len=10, head_dim=64]
        let q = Tensor::randn(0.0f32, 1.0f32, (2, 4, 10, 64), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0f32, (2, 4, 10, 64), &device).unwrap();

        let (q_embed, k_embed) = rope.forward(&q, &k, None).unwrap();

        assert_eq!(q_embed.dims(), q.dims());
        assert_eq!(k_embed.dims(), k.dims());
    }

    #[test]
    fn test_rotate_half() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], (1, 4), &device).unwrap();

        let rotated = SundialRotaryEmbedding::rotate_half(&x).unwrap();
        let result: Vec<Vec<f32>> = rotated.to_vec2().unwrap();

        // Should be: [-3, -4, 1, 2]
        assert!((result[0][0] - (-3.0)).abs() < 1e-5);
        assert!((result[0][1] - (-4.0)).abs() < 1e-5);
        assert!((result[0][2] - 1.0).abs() < 1e-5);
        assert!((result[0][3] - 2.0).abs() < 1e-5);
    }
}
