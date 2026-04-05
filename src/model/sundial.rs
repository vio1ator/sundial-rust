//! Main Sundial model for time series forecasting
//!
//! Integrates the transformer backbone with flow matching for probabilistic forecasting.
//! Matches the SundialForPrediction from the Python implementation.

use crate::flow::network::SimpleMLPAdaLN;
use crate::flow::sampling::flow_sample;
use crate::model::config::SundialConfig;
use crate::model::transformer::SundialTransformer;
use candle_core::{Device, Result, Tensor};
use candle_nn::Module;
use std::path::PathBuf;

/// Sundial model for prediction
pub struct SundialModel {
    config: SundialConfig,
    model: SundialTransformer,
    flow_loss: SimpleMLPAdaLN,
}

impl SundialModel {
    /// Create a new Sundial model
    pub fn new(config: &SundialConfig, vb: candle_nn::VarBuilder) -> Result<Self> {
        let model = SundialTransformer::new(config, vb.pp("model"))?;

        // Flow loss network
        let flow_loss = SimpleMLPAdaLN::new(
            config.output_token_lens[0],
            config.hidden_size,
            config.output_token_lens[0],
            config.hidden_size,
            config.flow_loss_depth,
            vb.pp("flow_loss"),
        )?;

        Ok(Self {
            config: config.clone(),
            model,
            flow_loss,
        })
    }

    /// Apply Reversible Instance Normalization (RevIN)
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq_len]
    ///
    /// # Returns
    /// Normalized tensor, mean, and std
    pub fn revin_normalize(x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let _batch = x.dim(0)?;

        // Calculate mean: [batch]
        let means = x.mean_keepdim(1)?;

        // Calculate std: [batch] (unbiased=False like Python)
        let stdev = ((x.powf(2.0)?).mean_keepdim(1)? - means.powf(2.0)?)?.sqrt()?;

        // Add small value to avoid division by zero (same as Python: +1e-5)
        let stdev = stdev.add(&Tensor::full(1e-5f32, stdev.shape(), stdev.device())?)?;

        // Normalize: (x - mean) / std
        // Expand means and stdev for broadcasting
        let means_exp = means.broadcast_add(&Tensor::zeros(x.shape(), x.dtype(), x.device())?)?;
        let stdev_exp = stdev.broadcast_mul(&Tensor::ones(x.shape(), x.dtype(), x.device())?)?;
        let normalized = x.sub(&means_exp)?.div(&stdev_exp)?;

        Ok((normalized, means, stdev))
    }

    /// Denormalize predictions using RevIN
    pub fn revin_denormalize(
        predictions: &Tensor,
        stdev: &Tensor,
        means: &Tensor,
    ) -> Result<Tensor> {
        predictions.mul(stdev)?.add(means)
    }

    /// Generate forecasts
    ///
    /// # Arguments
    /// * `input_ids` - Input time series of shape [batch_size, seq_len]
    /// * `max_new_tokens` - Number of future timesteps to forecast
    /// * `num_samples` - Number of probabilistic samples to generate
    /// * `revin` - Whether to apply RevIN normalization
    ///
    /// # Returns
    /// Forecasts of shape [num_samples, batch_size, forecast_length]
    pub fn generate(
        &self,
        input_ids: &Tensor,
        max_new_tokens: usize,
        num_samples: usize,
        revin: bool,
    ) -> Result<Tensor> {
        let batch_size = input_ids.dim(0)?;

        // Apply RevIN if requested
        let (input_ids, means, stdev) = if revin {
            let (normalized, means, stdev) = Self::revin_normalize(input_ids)?;
            (normalized, Some(means), Some(stdev))
        } else {
            (input_ids.clone(), None, None)
        };

        // Encode input through transformer
        let hidden_states = self.model.forward(&input_ids)?;

        // Get the last hidden state for conditioning
        let (_, num_patches, hidden_size) = hidden_states.dims3()?;
        let last_hidden = hidden_states.narrow(1, num_patches - 1, 1)?;
        let last_hidden = last_hidden.reshape((batch_size, hidden_size))?;

        // Generate samples using flow matching
        let predictions = flow_sample(
            &self.flow_loss,
            &last_hidden,
            num_samples,
            self.config.output_token_lens[0],
            self.config.num_sampling_steps,
        )?;

        // Truncate to max_new_tokens if needed
        let actual_forecast_length = if max_new_tokens < self.config.output_token_lens[0] {
            max_new_tokens
        } else {
            self.config.output_token_lens[0]
        };
        let predictions = if max_new_tokens < self.config.output_token_lens[0] {
            predictions.narrow(2, 0, max_new_tokens)?
        } else {
            predictions
        };

        // Denormalize if RevIN was applied
        if revin {
            let stdev = stdev.unwrap();
            let means = means.unwrap();
            // predictions shape: [batch, num_samples, actual_forecast_length]
            // stdev/means shape: [batch, 1] (from mean_keepdim)
            // Squeeze to [batch], then reshape to [batch, 1, 1] for broadcasting
            let stdev_squeezed = stdev.squeeze(1)?; // [batch]
            let means_squeezed = means.squeeze(1)?; // [batch]
            let stdev_exp = stdev_squeezed.unsqueeze(1)?.unsqueeze(2)?; // [batch, 1, 1]
            let means_exp = means_squeezed.unsqueeze(1)?.unsqueeze(2)?; // [batch, 1, 1]
                                                                        // Expand to match predictions shape
            let stdev_expanded = stdev_exp.repeat(&[1, num_samples, actual_forecast_length])?;
            let means_expanded = means_exp.repeat(&[1, num_samples, actual_forecast_length])?;
            predictions.mul(&stdev_expanded)?.add(&means_expanded)
        } else {
            Ok(predictions)
        }
    }

    /// Get model configuration
    pub fn config(&self) -> &SundialConfig {
        &self.config
    }

    /// Get the transformer backbone
    pub fn transformer(&self) -> &SundialTransformer {
        &self.model
    }

    /// Load model from safetensors bytes in memory
    ///
    /// This allows loading weights without writing to disk.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `data` - Raw safetensors bytes (decompressed)
    /// * `device` - Device to load model on
    ///
    /// # Returns
    /// * `Ok(SundialModel)` with loaded weights
    /// * `Err(candle_core::Error)` if loading fails
    pub fn load_from_safetensors_bytes(
        config: SundialConfig,
        data: &[u8],
        device: &Device,
    ) -> candle_core::Result<Self> {
        use crate::model::loader::load_safetensors_from_bytes;

        let tensors = load_safetensors_from_bytes(data, device)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load safetensors from bytes: {}", e)))?;
        tracing::info!("Loaded {} tensors from memory", tensors.len());

        // Create VarBuilder from loaded tensors with proper mapping
        let vb = crate::model::loader::create_varbuilder(tensors, device)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create varbuilder: {}", e)))?;

        tracing::info!("Creating SundialModel with loaded weights from memory...");
        SundialModel::new(&config, vb)
    }

    /// Load model from safetensors file
    ///
    /// Automatically resolves the model path using WeightLoader, which checks:
    /// 1. SUNDIAL_MODEL_PATH environment variable for external weights
    /// 2. Extracts embedded compressed weights to temporary storage
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `path` - Path to model weights (can be override with SUNDIAL_MODEL_PATH env var)
    /// * `device` - Device to load model on
    ///
    /// # Returns
    /// * `Ok(SundialModel)` with loaded weights
    /// * `Err(candle_core::Error)` if loading fails
    pub fn load_from_safetensors<P: AsRef<std::path::Path>>(
        config: SundialConfig,
        path: P,
        device: &Device,
    ) -> candle_core::Result<Self> {
        use crate::model::loader::{create_varbuilder, load_safetensors};
        use crate::weights::WeightLoader;
        use std::env;

        let path_ref = path.as_ref();

        // Check if SUNDIAL_MODEL_PATH environment variable is set
        let final_path = if let Ok(external_path) = env::var("SUNDIAL_MODEL_PATH") {
            tracing::info!(
                "Using external weights from SUNDIAL_MODEL_PATH: {:?}",
                external_path
            );
            PathBuf::from(external_path)
        } else if path_ref.exists() {
            // Use provided path if it exists
            tracing::info!("Using provided model path: {:?}", path_ref);
            path_ref.to_path_buf()
        } else {
            // Fall back to WeightLoader for embedded weights
            tracing::info!(
                "Model path {:?} not found, using embedded weights via WeightLoader",
                path_ref
            );
            WeightLoader::new()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to create weight loader: {}", e))
                })?
                .model_path()
                .to_path_buf()
        };

        tracing::info!("Loading weights from {:?}", final_path);

        let tensors = load_safetensors(&final_path, device)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load safetensors: {}", e)))?;
        tracing::info!("Loaded {} tensors from safetensors", tensors.len());

        // Create VarBuilder from loaded tensors with proper mapping
        let vb = create_varbuilder(tensors, device)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create varbuilder: {}", e)))?;

        tracing::info!("Creating SundialModel with loaded weights...");
        SundialModel::new(&config, vb)
    }
}

impl Module for SundialModel {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Simple forward for encoding only
        self.model.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_sundial_model_creation() {
        let config = SundialConfig {
            input_token_len: 16,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            output_token_lens: vec![32],
            flow_loss_depth: 2,
            ..Default::default()
        };

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);

        let model = SundialModel::new(&config, vb);
        assert!(model.is_ok());
    }

    #[test]
    fn test_revinnormalize() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
            (2, 3),
            &device,
        )
        .unwrap();

        let (normalized, means, stdev) = SundialModel::revin_normalize(&x).unwrap();

        // Check means: [2.0, 5.0]
        let means_vec: Vec<Vec<f32>> = means.to_vec2().unwrap();
        assert!((means_vec[0][0] - 2.0).abs() < 1e-5);
        assert!((means_vec[1][0] - 5.0).abs() < 1e-5);

        // Check normalized has mean ~0
        let norm_mean = normalized.mean_keepdim(1).unwrap();
        let norm_mean_vec: Vec<Vec<f32>> = norm_mean.to_vec2().unwrap();
        assert!(norm_mean_vec[0][0].abs() < 1e-5);
        assert!(norm_mean_vec[1][0].abs() < 1e-5);
    }

    #[test]
    fn test_generate_shape() {
        let config = SundialConfig {
            input_token_len: 16,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            output_token_lens: vec![32],
            flow_loss_depth: 2,
            num_sampling_steps: 5,
            ..Default::default()
        };

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);
        let model = SundialModel::new(&config, vb).unwrap();

        // Input: [batch=2, seq_len=64] (4 patches)
        let input = Tensor::randn(0.0f32, 1.0f32, (2, 64), &device).unwrap();

        // Generate 3 samples of 32 timesteps
        let output = model.generate(&input, 32, 3, false).unwrap();

        assert_eq!(output.dims(), &[2, 3, 32]); // [batch, num_samples, forecast_length]
    }
}
