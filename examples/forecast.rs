//! Example: Run Sundial forecasting from command line
//!
//! This example demonstrates how to use the Sundial model for time series forecasting.
//! It can load pretrained weights from HuggingFace or use random initialization.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use sundial_rust::{SundialConfig, SundialModel};

#[derive(Parser, Debug)]
#[command(name = "sundial-forecast")]
#[command(about = "Time series forecasting with Sundial model")]
struct Args {
    /// Ticker symbol (e.g., SPY)
    #[arg(short, long, default_value = "SPY")]
    ticker: String,

    /// Forecast length (number of timesteps)
    #[arg(short, long, default_value = "14")]
    forecast_length: usize,

    /// Number of samples to generate
    #[arg(short, long, default_value = "100")]
    num_samples: usize,

    /// Input sequence length
    #[arg(short, long, default_value = "2880")]
    seq_len: usize,

    /// Path to safetensors file (optional)
    #[arg(long)]
    model_path: Option<String>,

    /// Use random weights (for testing)
    #[arg(long, default_value = "false")]
    random_weights: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Sundial Time Series Forecasting");
    println!("================================");
    println!("Ticker: {}", args.ticker);
    println!("Forecast Length: {} timesteps", args.forecast_length);
    println!("Number of Samples: {}", args.num_samples);
    println!();

    // Load model configuration
    let config = SundialConfig::default();
    println!("Model Configuration:");
    println!("  Hidden Size: {}", config.hidden_size);
    println!("  Num Layers: {}", config.num_hidden_layers);
    println!("  Num Heads: {}", config.num_attention_heads);
    println!("  Patch Length: {}", config.input_token_len);
    println!();

    // Initialize model
    let device = Device::Cpu;

    let model = if args.random_weights {
        println!("Creating model with random weights...");
        let vb = VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);
        SundialModel::new(&config, vb)?
    } else if let Some(model_path) = args.model_path {
        println!("Loading model from {:?}...", model_path);
        SundialModel::load_from_safetensors(config, &model_path, &device)?
    } else {
        println!("No model path specified, using random weights for demonstration...");
        println!("To load actual pretrained weights, use --model-path <path>");
        println!();
        let vb = VarBuilder::from_varmap(&candle_nn::VarMap::new(), DType::F32, &device);
        SundialModel::new(&config, vb)?
    };

    println!("Model loaded successfully!");
    println!();

    // Create dummy input (in production, load real data)
    println!("Preparing input data...");
    let input = Tensor::randn(0.0f32, 1.0f32, (1, args.seq_len), &device)?;
    println!("Input shape: {:?}", input.dims());
    println!();

    // Generate forecasts
    println!("Generating forecasts...");
    let start = std::time::Instant::now();
    let forecasts = model.generate(&input, args.forecast_length, args.num_samples, true)?;
    let elapsed = start.elapsed();

    println!("Forecast shape: {:?}", forecasts.dims());
    println!("Generation time: {:.2?}", elapsed);
    println!();

    // Print summary statistics
    let mean_forecast = forecasts.mean_keepdim(1)?.squeeze(1)?; // [batch, forecast_length]
    let std_forecast = forecasts.var_keepdim(1)?.sqrt()?.squeeze(1)?; // [batch, forecast_length]

    println!("Forecast Statistics (averaged over samples):");
    println!("  Mean shape: {:?}", mean_forecast.dims());
    println!("  Std shape: {:?}", std_forecast.dims());
    println!();

    // Print first few forecast values
    println!("First 5 forecast values (mean across samples):");
    let mean_squeezed = mean_forecast.squeeze(0)?; // [14]
    for i in 0..5.min(args.forecast_length) {
        let val: f32 = mean_squeezed.get(i)?.to_scalar()?;
        println!("  Day {}: {:.4}", i + 1, val);
    }

    println!();
    println!("Forecasting complete!");

    Ok(())
}
