//! Reference tensor loading utilities
//!
//! This module provides functions to load Python-generated reference tensors
//! from .npy files for comparison with Rust implementations.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use ndarray::{ArrayD, Axis};
use std::path::Path;

/// Load a tensor from a .npy file
///
/// # Arguments
/// * `path` - Path to the .npy file
///
/// # Returns
/// A candle Tensor loaded from the file
pub fn load_reference_tensor<P: AsRef<Path>>(path: P) -> Result<Tensor> {
    let path = path.as_ref();
    
    if !path.exists() {
        bail!("Reference tensor file not found: {}", path.display());
    }
    
    // Load using ndarray
    let array: ArrayD<f32> = ndarray_npy::read_npy(path)?;
    
    // Convert to candle Tensor
    let shape: &[usize] = array.shape();
    let vec: Vec<f32> = array.iter().cloned().collect();
    
    let device = Device::Cpu;
    let tensor = Tensor::from_vec(vec, shape, &device)?;
    
    Ok(tensor)
}

/// Load multiple reference tensors from a directory
///
/// # Arguments
/// * `dir_path` - Path to directory containing .npy files
///
/// # Returns
/// A HashMap of tensor name -> Tensor
pub fn load_reference_tensors_from_dir<P: AsRef<Path>>(
    dir_path: P,
) -> Result<std::collections::HashMap<String, Tensor>> {
    let dir = dir_path.as_ref();
    
    if !dir.is_dir() {
        bail!("Reference directory not found: {}", dir.display());
    }
    
    let mut tensors = std::collections::HashMap::new();
    
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().map_or(false, |ext| ext == "npy") {
            if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                let tensor = load_reference_tensor(&path)?;
                tensors.insert(name.to_string(), tensor);
            }
        }
    }
    
    Ok(tensors)
}

/// Load a specific tensor by name from a directory
///
/// # Arguments
/// * `dir_path` - Path to directory containing .npy files
/// * `name` - Name of the tensor (without .npy extension)
///
/// # Returns
/// The requested Tensor
pub fn load_tensor_by_name<P: AsRef<Path>>(
    dir_path: P,
    name: &str,
) -> Result<Tensor> {
    let path = dir_path.as_ref().join(format!("{}.npy", name));
    load_reference_tensor(path)
}

/// Save a tensor to a .npy file (for debugging)
///
/// # Arguments
/// * `tensor` - Tensor to save
/// * `path` - Output path
pub fn save_tensor_to_npy<P: AsRef<Path>>(tensor: &Tensor, path: P) -> Result<()> {
    let path = path.as_ref();
    
    // Convert candle Tensor to ndarray
    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    
    let array = ndarray::Array::from_shape_vec(shape, data)?;
    ndarray_npy::write_npy(path, &array)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use tempfile::TempDir;

    #[test]
    fn test_save_and_load() -> Result<()> {
        let device = Device::Cpu;
        
        // Create a test tensor
        let original = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2, 3),
            &device,
        )?;
        
        // Save to temp file
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path().join("test.npy");
        
        save_tensor_to_npy(&original, &path)?;
        
        // Load it back
        let loaded = load_reference_tensor(&path)?;
        
        // Verify
        let orig_vec: Vec<f32> = original.flatten_all()?.to_vec1()?;
        let loaded_vec: Vec<f32> = loaded.flatten_all()?.to_vec1()?;
        
        assert_eq!(orig_vec, loaded_vec);
        
        Ok(())
    }
}
