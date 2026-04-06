//! End-to-end comparison: Same CSV input, compare final predictions
//!
//! This example loads the same CSV data that was used for Python,
//! runs inference with the Rust Sundial model, and compares the predictions.
//!
//! Usage:
//!     cargo run --release --example compare_e2e_predictions

use candle_core::{Device, Tensor};
use std::fs;
use std::io::{Read, Write};

fn load_csv_to_tensor(path: &str) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mut file = std::fs::File::open(path)?;
    let mut buffer = String::new();
    file.read_to_string(&mut buffer)?;

    // Skip header line
    let lines: Vec<&str> = buffer.lines().skip(1).collect();

    let values: Vec<f32> = lines
        .iter()
        .map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            parts[1].trim().parse::<f32>().unwrap_or(0.0)
        })
        .collect();

    println!("Loaded {} values from CSV", values.len());
    let len = values.len();
    Ok(Tensor::from_vec(values, (1, len), &Device::Cpu)?)
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
        "[{}] {}: shape={:?}, mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
        name,
        name,
        tensor.dims(),
        mean,
        std,
        min,
        max
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Sundial Rust vs PyTorch - End-to-End Prediction Comparison");
    println!("===========================================================\n");

    // Load the same CSV data
    println!("Loading test data from /tmp/e2e_test_data.csv...");
    let input = load_csv_to_tensor("/tmp/e2e_test_data.csv")?;
    debug_tensor("input", &input);

    // Load config and model
    let config = sundial_rust::SundialConfig::default();
    let device = Device::Cpu;

    println!("\nLoading Sundial model...");
    let model = sundial_rust::SundialModel::load_from_safetensors(
        config,
        "weights/model.safetensors",
        &device,
    )?;
    println!("Model loaded successfully!\n");

    // Run inference with RevIN disabled (to match Python)
    println!("Running inference (revin=false)...");
    let start = std::time::Instant::now();
    let output = model.generate(&input, 14, 1, false)?;
    let elapsed = start.elapsed();

    println!("\nOutput shape: {:?}", output.dims());
    println!("Generation time: {:.2}s\n", elapsed.as_secs_f64());

    // Extract predictions (squeeze batch and sample dimensions)
    let predictions = output.squeeze(0)?.squeeze(0)?;
    debug_tensor("predictions", &predictions);

    // Save Rust predictions
    let mut file = fs::File::create("/tmp/rust_predictions.csv")?;
    writeln!(file, "step,prediction")?;
    for i in 0..predictions.dim(0)? {
        let val: f32 = predictions.get(i)?.to_scalar()?;
        writeln!(file, "{}, {:.6}", i + 1, val)?;
    }
    println!("\nSaved Rust predictions to /tmp/rust_predictions.csv");

    // Print predictions
    println!("\nRust predictions (first 14):");
    for i in 0..14.min(predictions.dim(0)?) {
        let val: f32 = predictions.get(i)?.to_scalar()?;
        println!("  Step {}: {:.6}", i + 1, val);
    }

    Ok(())
}
