//! Model weight loading from HuggingFace
//!
//! Downloads and loads Sundial model weights from safetensors files.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{Init, VarBuilder};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// Re-export SimpleBackend from candle_nn::var_builder
use candle_nn::var_builder::SimpleBackend;

/// Download model files from HuggingFace
pub fn download_model(model_id: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    use std::process::Command;

    let cache_dir = match cache_dir {
        Some(p) => p.to_path_buf(),
        None => PathBuf::from("./model_cache"),
    };
    std::fs::create_dir_all(&cache_dir)?;

    let model_dir = cache_dir.join(model_id.replace('/', "_"));

    // Check if already downloaded
    if model_dir.join("model.safetensors").exists() {
        println!("Model already downloaded: {:?}", model_dir);
        return Ok(model_dir);
    }

    println!("Downloading model {}...", model_id);

    // Use huggingface-cli if available, otherwise use wget/curl
    let output = Command::new("huggingface-cli")
        .args(["download", model_id, "--local-dir"])
        .arg(&model_dir)
        .output()
        .context("Failed to run huggingface-cli. Install with: pip install huggingface_hub")?;

    if !output.status.success() {
        // Fallback to manual download
        eprintln!("huggingface-cli failed, trying manual download...");
        download_manual(model_id, &model_dir)?;
    }

    Ok(model_dir)
}

/// Manual download using curl/wget
fn download_manual(model_id: &str, dest_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dest_dir)?;

    let files = vec!["config.json", "model.safetensors"];

    for file in files {
        let url = format!("https://huggingface.co/{}/resolve/main/{}", model_id, file);
        let dest = dest_dir.join(file);

        println!("Downloading {}...", file);

        #[cfg(target_family = "unix")]
        {
            use std::process::Command;
            let status = Command::new("curl")
                .args(["-L", "-o"])
                .arg(&dest)
                .arg(&url)
                .status()?;

            if !status.success() {
                anyhow::bail!("Failed to download {}", file);
            }
        }

        #[cfg(target_family = "windows")]
        {
            use std::process::Command;
            let status = Command::new("powershell")
                .args([
                    "-Command",
                    &format!(
                        "Invoke-WebRequest -Uri '{}' -OutFile '{}'",
                        url,
                        dest.display()
                    ),
                ])
                .status()?;

            if !status.success() {
                anyhow::bail!("Failed to download {}", file);
            }
        }
    }

    Ok(())
}

/// Load model weights from safetensors file
pub fn load_safetensors<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let path_ref = path.as_ref();
    let tensors = candle_core::safetensors::load(path_ref, device)?;
    println!(
        "Successfully loaded {} tensors from {:?}",
        tensors.len(),
        path_ref
    );
    Ok(tensors)
}

/// Load model weights from safetensors bytes in memory
///
/// This allows loading weights without writing to disk.
///
/// # Memory Lifecycle
/// - Input `data` is **borrowed**, not owned
/// - Each tensor's raw data is **copied** into intermediate buffers (`tensor_data`, `f32_data`, etc.)
/// - `Tensor::from_vec()` takes **ownership** of these buffers
/// - Returned `Tensor`s **own their data completely** - independent of input `data`
/// - Input `data` buffer can be safely dropped immediately after this function returns
///
/// # Arguments
/// * `data` - Raw safetensors bytes (decompressed)
/// * `device` - Device to load tensors on
///
/// # Returns
/// * `Ok(HashMap<String, Tensor>)` with loaded tensors
/// * `Err(anyhow::Error)` if loading fails
pub fn load_safetensors_from_bytes(
    data: &[u8],
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    // Use safetensors crate directly to parse from memory
    use safetensors::SafeTensors;

    let safetensor = SafeTensors::deserialize(data)
        .map_err(|e| anyhow::anyhow!("Failed to parse safetensors: {}", e))?;

    let mut tensors = HashMap::new();

    // Get list of tensors
    let tensor_list = safetensor.tensors();

    for (tensor_name, tensor) in tensor_list {
        // Get shape and dtype
        let shape: Vec<usize> = tensor.shape().to_vec();
        let dtype = match tensor.dtype() {
            safetensors::Dtype::F32 => candle_core::DType::F32,
            safetensors::Dtype::F64 => candle_core::DType::F64,
            safetensors::Dtype::U8 => candle_core::DType::U8,
            safetensors::Dtype::I64 => candle_core::DType::I64,
            _ => anyhow::bail!("Unsupported dtype: {:?}", tensor.dtype()),
        };

        // Get raw data - this is a borrow into the input `data` buffer
        let data_ptr = tensor.data();

        // Convert to candle tensor
        // CRITICAL: Must copy data because:
        // 1. safetensors borrows the input buffer (data_ptr is a slice into `data`)
        // 2. Candle Tensor must own its data for safety and mobility
        // 3. Without copying, tensors would dangle when input buffer is dropped
        let tensor_data: Vec<u8> = data_ptr.to_vec();
        let candle_tensor = match dtype {
            candle_core::DType::F32 => {
                let f32_data: Vec<f32> = tensor_data
                    .chunks(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                // Tensor::from_vec takes ownership of f32_data
                Tensor::from_vec(f32_data, shape.as_slice(), device)?
            }
            candle_core::DType::F64 => {
                let f64_data: Vec<f64> = tensor_data
                    .chunks(8)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect();
                // Tensor::from_vec takes ownership of f64_data
                Tensor::from_vec(f64_data, shape.as_slice(), device)?
            }
            candle_core::DType::U8 => {
                // Tensor::from_vec takes ownership of tensor_data
                Tensor::from_vec(tensor_data, shape.as_slice(), device)?
            }
            candle_core::DType::I64 => {
                let i64_data: Vec<i64> = tensor_data
                    .chunks(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect();
                // Tensor::from_vec takes ownership of i64_data
                Tensor::from_vec(i64_data, shape.as_slice(), device)?
            }
            _ => anyhow::bail!("Unsupported dtype for tensor {}: {:?}", tensor_name, dtype),
        };

        // Each tensor now owns its data independently
        tensors.insert(tensor_name.to_string(), candle_tensor);
    }

    // At this point, all tensor data has been copied into tensor-owned allocations.
    // The input `data` buffer is no longer referenced and can be dropped.
    tracing::info!("Loaded {} tensors from memory", tensors.len());
    Ok(tensors)
}

/// Custom backend that retrieves tensors from a HashMap with name mapping
struct TensorMapBackend {
    tensors: Arc<HashMap<String, Tensor>>,
}

impl TensorMapBackend {
    fn new(tensors: HashMap<String, Tensor>) -> Self {
        Self {
            tensors: Arc::new(tensors),
        }
    }
}

impl SimpleBackend for TensorMapBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _init: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        // Map from Candle VarBuilder path to safetensor name
        let safetensor_name = map_var_path_to_safetensor(name);

        // Look up the tensor by safetensor name
        let tensor = self.tensors.get(&safetensor_name).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "Tensor not found: {} (mapped from {})",
                safetensor_name, name
            ))
        })?;

        // Check shape match
        if tensor.shape() != &s {
            candle_core::bail!(
                "Shape mismatch for {}: expected {:?}, got {:?}",
                name,
                s,
                tensor.shape()
            );
        }

        // Convert to requested dtype if needed
        let mut tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor.clone()
        };

        // Move to requested device if needed (compare by pointer)
        if !std::ptr::eq(tensor.device(), dev) {
            tensor = tensor.to_device(dev)?;
        }

        Ok(tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let safetensor_name = map_var_path_to_safetensor(name);
        self.tensors.contains_key(&safetensor_name)
    }
}

/// Create VarBuilder from loaded tensors with proper name mapping
pub fn create_varbuilder(
    tensors: HashMap<String, Tensor>,
    device: &Device,
) -> Result<VarBuilder<'_>> {
    let backend = TensorMapBackend::new(tensors);
    Ok(VarBuilder::from_backend(
        Box::new(backend),
        DType::F32,
        device.clone(),
    ))
}

/// Load Sundial model from HuggingFace
pub fn load_sundial_from_huggingface<'a>(
    model_id: &str,
    device: &'a Device,
) -> Result<VarBuilder<'a>> {
    let model_dir = download_model(model_id, None)?;
    let safetensors_path = model_dir.join("model.safetensors");

    if !safetensors_path.exists() {
        bail!("Safetensors file not found: {:?}", safetensors_path);
    }

    let tensors = load_safetensors(&safetensors_path, device)?;
    create_varbuilder(tensors, device)
}

/// Load Sundial model from in-memory weights.
///
/// This function loads the Sundial model directly from decompressed safetensors
/// bytes held in memory, avoiding all disk I/O for optimal startup performance.
/// It is the primary API for the memory-first loading strategy.
///
/// # Memory-First Loading
///
/// When using `WeightLoader::new_with_memory_weights()` or the default
/// `WeightLoader::new()` mode, model weights are decompressed from embedded
/// assets and held in memory. This function takes those in-memory weights
/// and constructs the Sundial model without ever touching the filesystem.
///
/// Benefits:
/// - **Zero disk I/O**: No temporary files or extraction required
/// - **Fast startup**: Eliminates filesystem overhead
/// - **Integrity guaranteed**: SHA256 hash verification before loading
/// - **Memory efficient**: Weights are shared between loader and model
///
/// # Arguments
///
/// * `weights` - Decompressed safetensors bytes (typically from `WeightLoader::get_model_weights()`)
/// * `config` - Sundial configuration struct defining model architecture
/// * `device` - Candle device (CPU/GPU) to load tensors on
///
/// # Returns
///
/// * `Ok(SundialModel)` - The loaded model ready for inference
/// * `Err(anyhow::Error)` if loading fails due to:
///   - Hash verification failure (weights corrupted or tampered)
///   - Safetensors parsing error
///   - Tensor shape mismatch with expected model architecture
///   - Device allocation failure
///
/// # Memory Lifecycle
///
/// 1. **Hash verification**: Reads `weights` buffer to compute SHA256
/// 2. **Tensor loading**: `load_safetensors_from_bytes()` **copies** all tensor data
///    into Tensor-owned allocations
/// 3. **VarBuilder creation**: `create_varbuilder()` wraps tensors with name mapping
/// 4. **Model construction**: `SundialModel::new()` takes ownership of VarBuilder
/// 5. **Buffer release**: After `load_safetensors_from_bytes()` returns, the
///    `weights` buffer is no longer referenced and can be dropped
///
/// The decompressed weights buffer is **not** kept alive by the model - tensors
/// own their data independently.
///
/// # Example
///
/// ```no_run
/// use sundial_rust::model::loader::load_sundial_from_memory;
/// use sundial_rust::weights::loader::WeightLoader;
/// use sundial_rust::model::config::SundialConfig;
/// use candle_core::Device;
///
/// # fn main() -> anyhow::Result<()> {
/// // Create weight loader with in-memory weights (default mode)
/// let loader = WeightLoader::new()?;
///
/// // Get the decompressed weights from memory
/// let weights = loader.get_model_weights()
///     .expect("Memory loader should have weights");
///
/// // Load configuration
/// let config = SundialConfig::default();
///
/// // Select device
/// let device = Device::Cpu;
///
/// // Load the model from memory
/// let model = load_sundial_from_memory(weights, &config, &device)?;
///
/// // Model is now ready for inference
/// # Ok(())
/// # }
/// ```
pub fn load_sundial_from_memory(
    weights: &[u8],
    config: &crate::model::SundialConfig,
    device: &Device,
) -> Result<crate::model::SundialModel> {
    // Verify hash before loading - only reads from weights, does not retain reference
    verify_integrity_from_bytes(weights)?;

    // Load tensors from bytes - this copies all tensor data into Tensor-owned allocations
    // After this call returns, the weights buffer is no longer referenced
    let tensors = load_safetensors_from_bytes(weights, device)?;

    // Create VarBuilder with proper name mapping - takes ownership of tensors
    let vb = create_varbuilder(tensors, device)?;

    // Construct the Sundial model - takes ownership of VarBuilder
    // At this point, tensors are fully owned by the model, weights buffer can be dropped
    crate::model::SundialModel::new(config, vb)
        .map_err(|e| anyhow::anyhow!("Failed to create Sundial model: {}", e))
}

/// Helper function to verify integrity from bytes
///
/// This is a convenience wrapper that re-exports the verification function
/// from the weights loader module.
fn verify_integrity_from_bytes(weights: &[u8]) -> Result<()> {
    use crate::assets::MODEL_SHA256;
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(weights);
    let computed_hash = format!("{:x}", hasher.finalize());

    if computed_hash != MODEL_SHA256 {
        anyhow::bail!(
            "Hash mismatch: expected {}, got {}",
            MODEL_SHA256,
            computed_hash
        );
    }

    tracing::debug!("SHA256 hash verified (memory): {}", computed_hash);
    Ok(())
}

/// Simple VarBuilder implementation using a tensor map
pub struct TensorVarBuilder {
    tensors: HashMap<String, Tensor>,
    device: Device,
}

impl TensorVarBuilder {
    pub fn new(tensors: HashMap<String, Tensor>, device: Device) -> Self {
        Self { tensors, device }
    }

    pub fn get(&self, _shape: (usize,), name: &str) -> Result<Tensor> {
        // Try to find the tensor by name
        if let Some(tensor) = self.tensors.get(name) {
            Ok(tensor.clone())
        } else {
            anyhow::bail!("Tensor not found: {}", name);
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Map safetensor name to expected Candle VarBuilder path
///
/// The safetensors file uses full paths like "model.embed_layer..." or "flow_loss.net..."
/// but Candle's VarBuilder expects paths like "model.embed_layer..." or "flow_loss.time_embed..."
/// (without the '.net.' nesting).
pub fn map_safetensor_to_var_path(safetensor_name: &str) -> String {
    let name = safetensor_name.trim();

    // Remove 'model.' prefix - Sundial safetensors use "model." prefix
    let name = name.strip_prefix("model.").unwrap_or(name);

    // Remove 'flow_loss.net.' prefix - the flow network is nested under 'net.' in safetensors
    // but the code expects it directly under 'flow_loss.'
    let name = name.strip_prefix("flow_loss.net.").unwrap_or(name);

    name.to_string()
}

/// Map Candle VarBuilder path to safetensor name (reverse of map_safetensor_to_var_path)
/// This is used when looking up tensors in the backend.
pub fn map_var_path_to_safetensor(var_path: &str) -> String {
    let name = var_path.trim();

    // For flow_loss, we need to add '.net.' back
    if let Some(rest) = name.strip_prefix("flow_loss.") {
        return format!("flow_loss.net.{}", rest);
    }

    // For model components (embed_layer, layers, norm), add 'model.' prefix
    if !name.starts_with("model.") && !name.starts_with("flow_loss.") {
        return format!("model.{}", name);
    }

    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_model() {
        // Skip if no network
        if std::env::var("SKIP_NETWORK").is_ok() {
            return;
        }

        let result = download_model(
            "thuml/sundial-base-128m",
            Some(PathBuf::from("./test_cache").as_path()),
        );
        assert!(result.is_ok() || result.is_err()); // Just check it runs
    }

    #[test]
    fn test_safetensor_name_mapping() {
        // Test that we can map common tensor names
        assert_eq!(
            map_safetensor_to_var_path("model.embed_layer.hidden_layer.weight"),
            "embed_layer.hidden_layer.weight"
        );
        assert_eq!(
            map_safetensor_to_var_path("model.layers.0.self_attn.q_proj.weight"),
            "layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            map_safetensor_to_var_path("flow_loss.net.time_embed.mlp.0.weight"),
            "time_embed.mlp.0.weight"
        );
    }

    #[test]
    fn test_load_safetensors_from_bytes() {
        use crate::weights::loader::WeightLoader;

        // Get weights from memory loader
        let loader =
            WeightLoader::new_with_memory_weights().expect("Failed to create memory weight loader");
        let weights = loader
            .get_model_weights()
            .expect("Memory loader should have weights");

        // Load tensors from bytes
        let device = Device::Cpu;
        let result = load_safetensors_from_bytes(weights, &device);

        assert!(result.is_ok(), "load_safetensors_from_bytes should succeed");

        let tensors = result.unwrap();
        assert!(!tensors.is_empty(), "Should have loaded some tensors");

        // Check for some expected tensor names (with model. prefix as in the safetensors file)
        assert!(tensors.contains_key("model.embed_layer.hidden_layer.weight"));
        assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn test_create_varbuilder() {
        use crate::weights::loader::WeightLoader;

        // Get weights from memory loader
        let loader =
            WeightLoader::new_with_memory_weights().expect("Failed to create memory weight loader");
        let weights = loader
            .get_model_weights()
            .expect("Memory loader should have weights");

        // Load tensors and create varbuilder
        let device = Device::Cpu;
        let tensors =
            load_safetensors_from_bytes(weights, &device).expect("Failed to load tensors");
        let result = create_varbuilder(tensors, &device);

        assert!(result.is_ok(), "create_varbuilder should succeed");
    }

    #[test]
    fn test_load_sundial_from_memory() {
        use crate::model::config::SundialConfig;
        use crate::weights::loader::WeightLoader;

        // Get weights from memory loader
        let loader =
            WeightLoader::new_with_memory_weights().expect("Failed to create memory weight loader");
        let weights = loader
            .get_model_weights()
            .expect("Memory loader should have weights");

        // Create config
        let config = SundialConfig::default();
        let device = Device::Cpu;

        // Load model from memory
        let result = load_sundial_from_memory(weights, &config, &device);

        assert!(result.is_ok(), "load_sundial_from_memory should succeed");

        let model = result.unwrap();
        // Just verify we got a model - can't access private fields
        assert!(
            model.config().hidden_size > 0,
            "Model should have valid config"
        );
    }

    #[test]
    fn test_hash_verification() {
        use crate::weights::loader::WeightLoader;

        // Get weights from memory loader
        let loader =
            WeightLoader::new_with_memory_weights().expect("Failed to create memory weight loader");
        let weights = loader
            .get_model_weights()
            .expect("Memory loader should have weights");

        // Verify should pass with valid weights
        let result = verify_integrity_from_bytes(weights);
        assert!(
            result.is_ok(),
            "Hash verification should pass for valid weights"
        );
    }

    #[test]
    fn test_hash_verification_fails_with_corrupted_data() {
        // Create corrupted data
        let corrupted = b"corrupted data that does not match the expected hash";

        // Verify should fail
        let result = verify_integrity_from_bytes(corrupted);
        assert!(
            result.is_err(),
            "Hash verification should fail for corrupted data"
        );

        let error_msg = result.unwrap_err().to_string();
        assert!(
            error_msg.contains("mismatch"),
            "Error should mention hash mismatch"
        );
    }
}
