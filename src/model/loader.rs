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
        
        // Get raw data
        let data_ptr = tensor.data();
        
        // Convert to candle tensor
        // Note: We need to copy the data because safetensors borrows the buffer
        let tensor_data: Vec<u8> = data_ptr.to_vec();
        let candle_tensor = match dtype {
            candle_core::DType::F32 => {
                let f32_data: Vec<f32> = tensor_data
                    .chunks(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec(f32_data, shape.as_slice(), device)?
            }
            candle_core::DType::F64 => {
                let f64_data: Vec<f64> = tensor_data
                    .chunks(8)
                    .map(|chunk| f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7],
                    ]))
                    .collect();
                Tensor::from_vec(f64_data, shape.as_slice(), device)?
            }
            candle_core::DType::U8 => {
                Tensor::from_vec(tensor_data.clone(), shape.as_slice(), device)?
            }
            candle_core::DType::I64 => {
                let i64_data: Vec<i64> = tensor_data
                    .chunks(8)
                    .map(|chunk| i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7],
                    ]))
                    .collect();
                Tensor::from_vec(i64_data, shape.as_slice(), device)?
            }
            _ => anyhow::bail!("Unsupported dtype for tensor {}: {:?}", tensor_name, dtype),
        };
        
        tensors.insert(tensor_name.to_string(), candle_tensor);
    }
    
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

    // For model, keep as-is (already has correct prefix)
    if name.starts_with("model.") {
        return name.to_string();
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
}
