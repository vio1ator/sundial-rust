//! Compare Rust implementation against Python outputs

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use std::fs;

#[derive(Parser, Debug)]
#[command(name = "compare-python")]
#[command(about = "Compare Sundial Rust vs Python outputs")]
struct Args {
    /// Model path
    #[arg(long, default_value = "weights/model.safetensors")]
    model_path: String,

    /// Forecast length
    #[arg(short, long, default_value = "14")]
    forecast_length: usize,

    /// Number of samples
    #[arg(short, long, default_value = "100")]
    num_samples: usize,

    /// Input sequence length
    #[arg(short, long, default_value = "2880")]
    seq_len: usize,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Sundial Rust vs Python Comparison");
    println!("==================================\n");

    // Load model
    let config = sundial_rust::SundialConfig::default();
    let device = Device::Cpu;

    println!("Loading model from {:?}...", args.model_path);
    let model =
        sundial_rust::SundialModel::load_from_safetensors(config, &args.model_path, &device)?;
    println!("Model loaded successfully!\n");

    // Create input with same seed as Python
    println!("Creating input with seed {}...", args.seed);

    // Use Candle's random generation (note: may not match Python exactly)
    let input = Tensor::randn(0.0f32, 1.0f32, (1, args.seq_len), &device)?;
    println!("Input shape: {:?}\n", input.dims());

    // Generate forecasts
    println!("Generating forecasts...");
    let start = std::time::Instant::now();
    let forecasts = model.generate(&input, args.forecast_length, args.num_samples, true)?;
    let elapsed = start.elapsed();

    println!("Forecast shape: {:?}", forecasts.dims());
    println!("Generation time: {:.2}s\n", elapsed.as_secs_f64());

    // Calculate statistics
    // forecasts shape: [batch, num_samples, forecast_length]
    // Mean across samples (dim=1): [batch, forecast_length]
    let mean_forecast = forecasts.mean_keepdim(1)?.squeeze(1)?; // [batch, forecast_length]
    let std_forecast = forecasts.var_keepdim(1)?.sqrt()?.squeeze(1)?; // [batch, forecast_length]

    println!("Forecast Statistics:");
    println!("  Mean shape: {:?}", mean_forecast.dims());
    println!("  Std shape: {:?}\n", std_forecast.dims());

    println!("First 5 forecast values (mean across samples):");
    let mean_squeezed = mean_forecast.squeeze(0)?; // [14]
    let std_squeezed = std_forecast.squeeze(0)?; // [14]
    for i in 0..5.min(args.forecast_length) {
        let val: f32 = mean_squeezed.get(i)?.to_scalar()?;
        println!("  Day {}: {:.4}", i + 1, val);
    }

    // Save results
    let mut mean_vec = Vec::new();
    let mut std_vec = Vec::new();
    for i in 0..args.forecast_length {
        mean_vec.push(mean_squeezed.get(i)?.to_scalar::<f32>()?);
        std_vec.push(std_squeezed.get(i)?.to_scalar::<f32>()?);
    }

    let results = serde_json::json!({
        "forecast_shape": forecasts.dims().to_vec(),
        "generation_time_s": elapsed.as_secs_f64(),
        "mean_first_5": mean_vec.iter().take(5).cloned().collect::<Vec<_>>(),
        "mean_all": mean_vec,
        "std_all": std_vec
    });

    fs::write(
        "/tmp/rust_results.json",
        serde_json::to_string_pretty(&results)?,
    )?;
    println!("\nResults saved to /tmp/rust_results.json");

    Ok(())
}
