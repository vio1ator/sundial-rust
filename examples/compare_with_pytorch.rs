//! Compare Rust Sundial with PyTorch implementation
//!
//! This example loads the actual Sundial weights and captures intermediate
//! tensors that can be compared with the Python implementation.
//!
//! Usage:
//!     SUNDIAL_DEBUG=1 cargo run --release --example compare_with_pytorch
//!
//! Then compare with Python outputs using:
//!     python scripts/compare_tensors.py --python /tmp/python_intermediates.npz --rust /tmp/

use candle_core::{DType, Device, Tensor};
use std::fs::File;
use std::io::Write;
use sundial_rust::{SundialConfig, SundialModel};

fn save_tensor_to_bin(name: &str, tensor: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(format!("/tmp/{}_rust.bin", name))?;
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

    // Write shape (4 bytes for dimension count + 4 bytes for each dimension)
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

fn debug_tensor(name: &str, tensor: &Tensor) {
    let mean = tensor.mean_all().unwrap().to_scalar::<f32>().unwrap();
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Sundial Rust vs PyTorch Comparison");
    println!("===================================\n");

    // Check if debug mode is enabled
    let debug_mode = std::env::var("SUNDIAL_DEBUG").is_ok();
    if !debug_mode {
        eprintln!("Warning: SUNDIAL_DEBUG environment variable not set.");
        eprintln!("Set it to enable debug output: SUNDIAL_DEBUG=1 cargo run --release --example compare_with_pytorch\n");
    }

    // Load config
    let config = SundialConfig::default();
    println!("Configuration:");
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Num layers: {}", config.num_hidden_layers);
    println!("  Num heads: {}", config.num_attention_heads);
    println!("  Head dim: {}", config.head_dim());
    println!("  Input token len: {}", config.input_token_len);
    println!("  Output token len: {}", config.output_token_lens[0]);
    println!();

    // Create device
    let device = Device::Cpu;

    // Load the actual model weights
    println!("Loading Sundial model from weights/model.safetensors...");
    let model = SundialModel::load_from_safetensors(config, "weights/model.safetensors", &device)?;
    println!("Model loaded successfully!\n");

    // Create random input (same as Python:.randn(1, 2880))
    // Note: Rust and PyTorch use different random seeds, so values won't match exactly
    // But the architecture and computations should produce equivalent results
    let batch_size = 1;
    let seq_len = 2880;
    println!("Creating random input: shape=[{}, {}]", batch_size, seq_len);
    let input = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len), &device)?;

    debug_tensor("input", &input);
    if debug_mode {
        save_tensor_to_bin("input", &input)?;
    }

    // Run inference with debug mode
    println!("\nRunning inference with revin=false...");
    let forecast_length = 14;
    let num_samples = 1;

    let start = std::time::Instant::now();
    let output = model.generate(&input, forecast_length, num_samples, false)?;
    let elapsed = start.elapsed();

    println!("\nOutput shape: {:?}", output.dims());
    println!("Generation time: {:.2}s", elapsed.as_secs_f64());

    if debug_mode {
        debug_tensor("final_output", &output);
        save_tensor_to_bin("final_output", &output)?;
    }

    // Print some sample values
    println!("\nFirst 10 forecast values (mean across samples):");
    let mean_forecast = output.mean_keepdim(0)?.squeeze(0)?.squeeze(0)?;
    for i in 0..10.min(forecast_length) {
        let val: f32 = mean_forecast.get(i)?.to_scalar()?;
        println!("  Step {}: {:.6}", i + 1, val);
    }

    println!("\nDebug outputs saved to /tmp/*.bin");
    println!("Use: python scripts/compare_tensors.py --python /tmp/python_intermediates.npz --rust /tmp/");

    Ok(())
}
