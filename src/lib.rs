//! Sundial Rust - Time Series Forecasting
//!
//! A Rust implementation of the Sundial time series forecasting model,
//! providing high-performance inference without Python dependencies.

pub mod assets;
pub mod data;
pub mod flow;
pub mod model;
pub mod testing;
pub mod weights;

pub use assets::*;
pub use data::{DataLoader, TimeSeriesData};
pub use model::{SundialConfig, SundialModel};
pub use weights::{get_config_path, get_model_path, WeightLoader};

/// Debug utilities for tensor inspection
pub mod debug_utils {
    use candle_core::{Result, Tensor};
    use std::fs::File;
    use std::io::Write;

    /// Print debug information for a tensor
    pub fn debug_tensor(name: &str, tensor: &Tensor) {
        let mean = tensor.mean_all().unwrap().to_scalar::<f32>().unwrap();
        // Calculate std manually since std_all is not available
        let mean_sq = tensor
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let std = (mean_sq - mean * mean).sqrt();
        let min = tensor.min_all().unwrap().to_scalar::<f32>().unwrap();
        let max = tensor.max_all().unwrap().to_scalar::<f32>().unwrap();

        println!(
            "[DEBUG] {}: shape={:?}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
            name,
            tensor.dims(),
            mean,
            std,
            min,
            max
        );
    }

    /// Save tensor to a simple binary format for comparison
    pub fn save_tensor_to_bin(name: &str, tensor: &Tensor) -> Result<()> {
        let mut file = File::create(format!("/tmp/{}_rust.bin", name))?;
        let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

        // Write shape (4 bytes for each dimension count + 4 bytes for dimension count)
        let dims = tensor.dims();
        let dim_count = dims.len() as u32;
        file.write_all(&dim_count.to_le_bytes())?;
        for &d in dims {
            file.write_all(&(d as u32).to_le_bytes())?;
        }

        // Write data
        for &val in &data {
            file.write_all(&val.to_le_bytes())?;
        }

        println!("[DEBUG] Saved {} to /tmp/{}_rust.bin", name, name);
        Ok(())
    }

    /// Load tensor from binary format
    pub fn load_tensor_from_bin(name: &str) -> Result<Tensor> {
        use candle_core::Error;
        use std::io::Read;

        let mut file = File::open(format!("/tmp/{}_python.bin", name))
            .map_err(|e| Error::Msg(format!("Failed to open file: {}", e)))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| Error::Msg(format!("Failed to read file: {}", e)))?;

        // Read shape
        let mut pos = 0;
        if buffer.len() < 4 {
            return Err(Error::Msg(
                "Buffer too short for dimension count".to_string(),
            ));
        }
        let dim_count = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]) as usize;
        pos += 4;

        let mut dims = Vec::with_capacity(dim_count);
        for _ in 0..dim_count {
            if pos + 4 > buffer.len() {
                return Err(Error::Msg("Buffer too short for dimensions".to_string()));
            }
            let d = u32::from_le_bytes([
                buffer[pos],
                buffer[pos + 1],
                buffer[pos + 2],
                buffer[pos + 3],
            ]) as usize;
            dims.push(d);
            pos += 4;
        }

        // Read data
        let data: Vec<f32> = buffer[pos..]
            .chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let device = candle_core::Device::Cpu;
        Ok(Tensor::from_vec(data, dims, &device)?)
    }

    /// Compare two tensors and print differences
    pub fn compare_tensors(name: &str, rust_tensor: &Tensor, python_tensor: &Tensor) -> Result<()> {
        let rust_data: Vec<f32> = rust_tensor.flatten_all()?.to_vec1()?;
        let python_data: Vec<f32> = python_tensor.flatten_all()?.to_vec1()?;

        if rust_data.len() != python_data.len() {
            println!(
                "[DEBUG] {}: SHAPE MISMATCH - Rust: {}, Python: {}",
                name,
                rust_data.len(),
                python_data.len()
            );
            return Ok(());
        }

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;
        let mut count = 0;

        for (r, p) in rust_data.iter().zip(python_data.iter()) {
            let diff = (r - p).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff;
            count += 1;
        }

        let mean_diff = sum_diff / count as f32;

        println!(
            "[DEBUG] {}: max_diff={:.8}, mean_diff={:.8}",
            name, max_diff, mean_diff
        );

        if max_diff > 1e-4 {
            println!("[DEBUG] {}: WARNING - Large differences detected!", name);
        }

        Ok(())
    }
}
