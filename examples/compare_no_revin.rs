//! Compare without RevIN

use anyhow::Result;
use candle_core::{Device, Tensor};
use clap::Parser;
use std::fs;

#[derive(Parser, Debug)]
#[command(name = "compare-no-revin")]
struct Args {
    #[arg(long, default_value = "weights/model.safetensors")]
    model_path: String,
    #[arg(short, long, default_value = "14")]
    forecast_length: usize,
    #[arg(short, long, default_value = "100")]
    num_samples: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Sundial Rust - No RevIN");
    println!("========================\n");

    let config = sundial_rust::SundialConfig::default();
    let device = Device::Cpu;

    let model =
        sundial_rust::SundialModel::load_from_safetensors(config, &args.model_path, &device)?;

    let input_data = load_numpy_f32("/tmp/input_data.npy")?;
    println!("Input shape: {:?}\n", input_data.dims());

    println!("Generating forecasts (revin=false)...");
    let start = std::time::Instant::now();
    let forecasts = model.generate(&input_data, args.forecast_length, args.num_samples, false)?;
    let elapsed = start.elapsed();

    println!("Forecast shape: {:?}", forecasts.dims());
    println!("Generation time: {:.2}s\n", elapsed.as_secs_f64());

    let mean_forecast = forecasts.mean_keepdim(0)?.squeeze(0)?.squeeze(0)?;

    println!("First 10 forecast values (mean across samples):");
    let mut mean_vec = Vec::new();
    for i in 0..10.min(args.forecast_length) {
        let val: f32 = mean_forecast.get(i)?.to_scalar()?;
        mean_vec.push(val);
        println!("  Day {}: {:.4}", i + 1, val);
    }

    let results = serde_json::json!({
        "forecast_shape": forecasts.dims().to_vec(),
        "generation_time_s": elapsed.as_secs_f64(),
        "mean_first_10": mean_vec,
    });

    fs::write(
        "/tmp/rust_no_revin.json",
        serde_json::to_string_pretty(&results)?,
    )?;

    Ok(())
}

fn load_numpy_f32(path: &str) -> Result<Tensor> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut header_end = 0;
    for i in 0..buffer.len() {
        if buffer[i] == b'\n' {
            header_end = i + 1;
            break;
        }
    }

    let data_bytes = &buffer[header_end..];
    let data: Vec<f32> = data_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(Tensor::from_vec(data, (1, 2880), &Device::Cpu)?)
}
