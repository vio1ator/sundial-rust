//! Sundial CLI - Time Series Forecasting Tool
//!
//! A portable CLI binary for loading and forecasting time series data.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use sundial_rust::{SundialConfig, SundialModel};
use tracing::info;

/// Sundial CLI - Time Series Forecasting Tool
#[derive(Parser, Debug)]
#[command(name = "sundial")]
#[command(author = "Sundial Team")]
#[command(version = "0.1.0")]
#[command(about = "Portable CLI tool for time series forecasting")]
#[command(long_about = "Sundial CLI - Time Series Forecasting Tool

A portable CLI tool for loading time series data and generating forecasts using pre-trained Candle ML models.

═══════════════════════════════════════════════════════════
📖 USAGE
═══════════════════════════════════════════════════════════

  Data exploration mode: Load and preview time series data
  Inference mode: Generate forecasts using a trained model

═══════════════════════════════════════════════════════════
📚 EXAMPLES
═══════════════════════════════════════════════════════════

  📊 Data Loading:
    # Load and preview time series data from CSV
    sundial --input data.csv

    # Load and preview time series data from Parquet
    sundial --input data.parquet

  🎯 Basic Forecasting:
    # Generate 10-step forecast with embedded weights (default)
    sundial --infer --input data.csv --horizon 10

    # Generate 10-step forecast with custom model
    sundial --infer --model weights/model.safetensors --input data.csv --horizon 10

  ⚙️ Custom Parameters:
    # Generate forecast with custom horizon (24 steps ahead)
    sundial --infer --model weights/model.safetensors --input data.csv --horizon 24

    # Use custom window size (60 historical points)
    sundial --infer --model weights/model.safetensors --input data.csv --window-size 60

    # Combine multiple custom parameters
    sundial --infer --model weights/model.safetensors --input data.csv --horizon 24 --window-size 60 --num-samples 100

  🔮 Uncertainty Estimation:
    # Generate predictions with uncertainty quantification (100 samples)
    sundial --infer --model weights/model.safetensors --input data.csv --horizon 10 --num-samples 100
    
    # This provides confidence intervals showing min/max/mean for each forecast step

  💾 Saving Output:
    # Save predictions to JSON format (using embedded weights)
    sundial --infer --input data.csv --output forecasts.json --format json

    # Save predictions to CSV format (with custom model)
    sundial --infer --model weights/model.safetensors --input data.csv --output forecasts.csv --format csv

  🤫 Quiet & Verbose Modes:
    # Quiet mode - only write to output file (useful for scripting)
    sundial --infer --model weights/model.safetensors --input data.csv --output forecasts.json --quiet

    # Verbose mode - show detailed information including intermediate steps
    sundial --infer --model weights/model.safetensors --input data.csv --horizon 10 --verbose

═══════════════════════════════════════════════════════════
📁 SUPPORTED INPUT FORMATS
═══════════════════════════════════════════════════════════

  • .csv      - Comma-separated values files
  • .parquet  - Apache Parquet columnar format

═══════════════════════════════════════════════════════════
📝 DATA REQUIREMENTS
═══════════════════════════════════════════════════════════

  Input files must contain at least these columns:

  • timestamp  - Time point identifier (any format accepted)
  • value      - Numeric time series value (float or integer)

═══════════════════════════════════════════════════════════
🔧 COMMON OPTIONS
═══════════════════════════════════════════════════════════

  -i, --input <FILE>       Input file path (CSV or Parquet) with time series data
  -o, --output <FILE>      Output file path for forecasts
  --horizon <N>            Number of future steps to forecast (default: 10)
  --window-size <N>        Number of historical points for input window (default: 30)
  --num-samples <N>        Number of stochastic samples for uncertainty (default: 1)
  -m, --model <FILE>       Path to pre-trained model (.safetensors)
  --format <FORMAT>        Output format: json or csv (default: json)
  --quiet                  Suppress console output (useful for scripting)
  --verbose                Show detailed/debug information
  --infer                  Run inference to generate forecasts (uses embedded weights by default)
  -m, --model <FILE>       Path to pre-trained model (.safetensors) to override embedded weights

═══════════════════════════════════════════════════════════
📋 QUICK REFERENCE
═══════════════════════════════════════════════════════════

  Basic usage: sundial --input <data_file>
  With forecasting (embedded weights): sundial --infer --input <data> --horizon <n>
  With custom model: sundial --infer --model <model> --input <data> --horizon <n>

  Run 'sundial --help' for full documentation.
")]
struct Args {
    /// Input file path (CSV or Parquet)
    #[arg(long, short)]
    input: Option<PathBuf>,

    /// Output file path (for forecasts)
    #[arg(long, short)]
    output: Option<PathBuf>,

    /// Number of steps to forecast
    #[arg(long, default_value = "10")]
    horizon: usize,

    /// Input window size for the model
    #[arg(long, default_value = "30")]
    window_size: usize,

    /// Number of samples to generate for uncertainty estimation
    #[arg(long, default_value = "1")]
    num_samples: usize,

    /// Model path (safetensors file) for inference
    #[arg(long, short)]
    model: Option<PathBuf>,

    /// Model configuration file (JSON)
    #[arg(long)]
    config: Option<PathBuf>,

    /// Output format (json or csv)
    #[arg(long, default_value = "json")]
    format: String,

    /// Quiet mode - suppress console output
    #[arg(long)]
    quiet: bool,

    /// Verbose output - show detailed information including intermediate steps
    #[arg(long, short, conflicts_with = "quiet")]
    verbose: bool,

    /// Run inference mode
    #[arg(long)]
    infer: bool,

    /// Show quick-start tutorial for first-time users
    #[arg(long)]
    quickstart: bool,

    /// Custom column name for timestamp
    #[arg(long, default_value = "timestamp")]
    timestamp_col: String,

    /// Custom column name for value
    #[arg(long, default_value = "value")]
    value_col: String,

    /// Disable auto-detection of column names
    #[arg(long, default_value = "false")]
    no_auto_detect: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Handle quickstart mode
    if args.quickstart {
        show_quickstart_guide();
        return Ok(());
    }

    // Validate column names are not empty
    if args.timestamp_col.is_empty() {
        anyhow::bail!("--timestamp-col cannot be empty");
    }
    if args.value_col.is_empty() {
        anyhow::bail!("--value-col cannot be empty");
    }

    // Store column names before moving them
    let timestamp_col = args.timestamp_col.clone();
    let value_col = args.value_col.clone();

    // Auto-detect column names if using defaults and auto-detection is enabled
    let (timestamp_col, value_col) = if !args.no_auto_detect {
        detect_column_names(&timestamp_col, &value_col)?
    } else {
        (timestamp_col, value_col)
    };

    // Handle inference mode
    let should_infer = args.infer || (args.output.is_some() && args.horizon > 0);

    if should_infer {
        if args.input.is_none() {
            anyhow::bail!("--input is required for inference/forecasting");
        }
        // Validate forecast parameters
        validate_forecast_params(&args)?;
        run_inference(&args, &timestamp_col, &value_col).await?;
    } else if let Some(input_path) = args.input {
        load_and_display_data(&input_path, &timestamp_col, &value_col, args.quiet).await?;
    } else {
        info!("No input file specified. Use --input to load data.");
        println!("Usage: sundial --input <file> [--output <output_file>] [--horizon <n>] [--window-size <n>]");
        println!("       sundial --infer --model <model> --input <data> [--horizon <n>] [--window-size <n>] [--num-samples <n>]");
        println!("Supported formats: .csv, .parquet");
        println!("\nRun 'sundial --quickstart' for an interactive tutorial.");
    }

    Ok(())
}

/// Validate forecast parameters against model requirements and logical constraints
fn validate_forecast_params(args: &Args) -> Result<()> {
    // Horizon must be positive
    if args.horizon == 0 {
        anyhow::bail!(
            "--horizon must be a positive integer (got {})",
            args.horizon
        );
    }

    // Horizon should be reasonable (not too large)
    if args.horizon > 1000 {
        anyhow::bail!(
            "--horizon value {} is too large. Maximum recommended value is 1000",
            args.horizon
        );
    }

    // Window size must be positive
    if args.window_size == 0 {
        anyhow::bail!(
            "--window-size must be a positive integer (got {})",
            args.window_size
        );
    }

    // Window size should be reasonable
    if args.window_size > 10000 {
        anyhow::bail!(
            "--window-size value {} is too large. Maximum recommended value is 10000",
            args.window_size
        );
    }

    // Number of samples must be positive
    if args.num_samples == 0 {
        anyhow::bail!(
            "--num-samples must be a positive integer (got {})",
            args.num_samples
        );
    }

    Ok(())
}

/// Load time series data from file (CSV or Parquet) and display information
async fn load_and_display_data(
    path: &PathBuf,
    timestamp_col: &str,
    value_col: &str,
    quiet: bool,
) -> Result<()> {
    // Check if file exists
    if !path.exists() {
        anyhow::bail!("Input file not found: {}", path.display());
    }

    // Determine file type by extension
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    let df = match extension.as_deref() {
        Some("csv") => load_csv(path, timestamp_col, value_col)?,
        Some("parquet") => load_parquet(path, timestamp_col, value_col)?,
        Some(other) => anyhow::bail!(
            "Unsupported file format: .{} (supported: .csv, .parquet)",
            other
        ),
        None => anyhow::bail!("Cannot determine file format from extension. Use .csv or .parquet"),
    };

    // Get column names
    let columns: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Display information
    if !quiet {
        println!(
            "Successfully loaded time series data from: {}",
            path.display()
        );
        println!("Row count: {}", df.height());
        println!("Column names: {:?}", columns);

        // Show sample data
        println!("\nFirst 5 rows:");
        let subset = df.head(Some(5));
        println!("{}", subset);
    }

    info!("Loaded {} rows with columns: {:?}", df.height(), columns);

    Ok(())
}

/// Load CSV file using csv crate and convert to polars DataFrame
fn load_csv(
    path: &PathBuf,
    timestamp_col: &str,
    value_col: &str,
) -> Result<polars::frame::DataFrame> {
    use polars::prelude::*;
    use std::io::BufReader;

    let file = std::fs::File::open(path).context("Failed to open CSV file")?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    // Get headers
    let headers: Vec<String> = csv_reader
        .headers()?
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Auto-detect column names if using defaults
    let (final_timestamp_col, final_value_col) =
        if timestamp_col == "timestamp" && value_col == "value" {
            // Try to auto-detect
            if let Some((detected_ts, detected_val)) = auto_detect_columns(&headers) {
                info!(
                    "Auto-detected columns: timestamp='{}', value='{}'",
                    detected_ts, detected_val
                );
                (detected_ts, detected_val)
            } else {
                // Fall back to defaults
                (timestamp_col.to_string(), value_col.to_string())
            }
        } else {
            // User explicitly specified, use as-is
            (timestamp_col.to_string(), value_col.to_string())
        };

    // Validate required columns
    for col in &[&final_timestamp_col, &final_value_col] {
        if !headers.iter().any(|h| h == *col) {
            anyhow::bail!(
                "Column '{}' not found in CSV. Available columns: {:?}",
                col,
                headers
            );
        }
    }

    // Read all records
    let mut records: Vec<Vec<String>> = Vec::new();
    for result in csv_reader.records() {
        let record = result?;
        records.push(record.iter().map(|s| s.to_string()).collect());
    }

    // Build columns
    let mut column_data: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for header in &headers {
        column_data.insert(header.clone(), Vec::new());
    }

    for record in &records {
        for (i, value) in record.iter().enumerate() {
            if let Some(col_name) = headers.get(i) {
                column_data.get_mut(col_name).unwrap().push(value.clone());
            }
        }
    }

    let col_vec = headers
        .iter()
        .map(|h| {
            polars::prelude::Column::from(polars::prelude::Series::new(
                h.clone().into(),
                column_data.get(h).unwrap(),
            ))
        })
        .collect::<Vec<_>>();

    Ok(polars::prelude::DataFrame::new(col_vec)?)
}

/// Load Parquet file using polars
fn load_parquet(
    path: &PathBuf,
    timestamp_col: &str,
    value_col: &str,
) -> Result<polars::frame::DataFrame> {
    use polars::prelude::*;

    let file = std::fs::File::open(path).context("Failed to open Parquet file")?;
    let mut reader = polars::prelude::ParquetReader::new(file);

    // Get schema to validate columns
    let schema = reader.schema()?;
    let column_names: Vec<String> = schema.iter_names().map(|s| s.to_string()).collect();

    // Auto-detect column names if using defaults
    let (final_timestamp_col, final_value_col) =
        if timestamp_col == "timestamp" && value_col == "value" {
            // Try to auto-detect
            if let Some((detected_ts, detected_val)) = auto_detect_columns(&column_names) {
                info!(
                    "Auto-detected columns: timestamp='{}', value='{}'",
                    detected_ts, detected_val
                );
                (detected_ts, detected_val)
            } else {
                // Fall back to defaults
                (timestamp_col.to_string(), value_col.to_string())
            }
        } else {
            // User explicitly specified, use as-is
            (timestamp_col.to_string(), value_col.to_string())
        };

    // Validate required columns
    for col in &[&final_timestamp_col, &final_value_col] {
        if !column_names.iter().any(|h| h == *col) {
            anyhow::bail!(
                "Column '{}' not found in Parquet file. Available columns: {:?}",
                col,
                column_names
            );
        }
    }

    // Read the entire file into a DataFrame
    let df = reader.finish()?;
    Ok(df)
}

/// Run inference with a pre-trained model
async fn run_inference(args: &Args, timestamp_col: &str, value_col: &str) -> Result<()> {
    use candle_core::{Device, Tensor};
    use std::fs;

    let device = Device::Cpu;

    // Load model configuration
    let config = if let Some(config_path) = &args.config {
        let config_str = fs::read_to_string(config_path)?;
        serde_json::from_str(&config_str).context("Failed to parse model config")?
    } else {
        // Use default Sundial base-128m configuration
        SundialConfig::default()
    };

    // Determine model path: check --model flag first, then use WeightLoader
    let (model_path, _weight_loader_guard) = if let Some(ref model_path) = args.model {
        info!("Using model path from --model flag: {:?}", model_path);

        // Validate that the model file exists
        if !model_path.exists() {
            anyhow::bail!(
                "Model file not found: {}\n\nUsage:\n  - Use --model with a valid path to a .safetensors file\n  - Omit --model to use embedded weights (default behavior)\n  - Check available model paths in the README",
                model_path.display()
            );
        }

        if !model_path.is_file() {
            anyhow::bail!(
                "Model path is not a file: {}\n\nUsage:\n  - --model must point to a .safetensors model file\n  - Omit --model to use embedded weights (default behavior)",
                model_path.display()
            );
        }

        (model_path.clone(), None)
    } else {
        // No --model flag provided, use WeightLoader to get embedded weights
        // Keep the loader alive to prevent temp directory cleanup
        info!("No --model flag provided, using embedded weights via WeightLoader");
        use sundial_rust::WeightLoader;
        let loader = WeightLoader::new_with_verbose(args.verbose)
            .context("Failed to create weight loader")?;
        info!("Embedded weights path: {:?}", loader.model_path());
        (loader.model_path().to_path_buf(), Some(loader))
    };

    info!("Loading model from {:?}", model_path);

    // Load the model from safetensors
    let model = SundialModel::load_from_safetensors(config.clone(), &model_path, &device)?;

    info!(
        "Model loaded successfully with config: hidden_size={}, layers={}",
        config.hidden_size, config.num_hidden_layers
    );

    // Load input data with auto-detected column names
    let input_path = args.input.as_ref().unwrap();
    let df = if input_path.extension().and_then(|e| e.to_str()) == Some("csv") {
        load_csv(input_path, timestamp_col, value_col)?
    } else {
        load_parquet(input_path, timestamp_col, value_col)?
    };

    // Extract time series values using configured column names
    let values_col = df.column(value_col)?;
    // Cast to f32 if needed (handles string or integer input)
    let values_series = if values_col.dtype().is_float() {
        values_col.clone()
    } else {
        values_col.cast(&polars::prelude::DataType::Float32)?
    };
    let values: Vec<f32> = values_series
        .f32()?
        .into_iter()
        .map(|opt| opt.unwrap_or(0.0))
        .collect();

    info!("Loaded {} data points", values.len());

    // Prepare input window - model expects input_token_len patches
    let patch_size = config.input_token_len;
    let required_window = patch_size;

    if values.len() < required_window {
        anyhow::bail!(
            "Input data has {} points, but model requires at least {} points (input_token_len)",
            values.len(),
            required_window
        );
    }

    let window_size = args.window_size.min(values.len());
    let input_window: Vec<f32> = values[values.len() - window_size..].to_vec();
    let input_window_len = input_window.len(); // Store length for verbose output

    // Create input tensor [batch=1, seq_len]
    let input_tensor = Tensor::from_vec(input_window.clone(), (1, window_size), &device)?;

    if args.verbose {
        info!(
            "Using timestamp column: '{}', value column: '{}'",
            args.timestamp_col, args.value_col
        );
    }
    info!("Input tensor shape: {:?}", input_tensor.shape());

    info!(
        "Running inference with horizon={}, window_size={}, num_samples={}",
        args.horizon, args.window_size, args.num_samples
    );

    // Generate forecasts
    // Note: model.generate returns [batch, num_samples, forecast_length]
    let predictions = model.generate(&input_tensor, args.horizon, args.num_samples, true)?;
    info!("Predictions shape: {:?}", predictions.shape());

    // Extract predictions and compute statistics
    // Shape after squeeze: [num_samples, forecast_length]
    let pred_2d = predictions.squeeze(0)?.to_vec2()?;
    let num_samples = pred_2d.len();
    let forecast_length = if num_samples > 0 { pred_2d[0].len() } else { 0 };

    // Compute mean, min, max for each forecast step
    let mut means = vec![0.0f32; forecast_length];
    let mut mins = vec![f32::INFINITY; forecast_length];
    let mut maxs = vec![f32::NEG_INFINITY; forecast_length];

    for sample in &pred_2d {
        for (i, &val) in sample.iter().enumerate() {
            means[i] += val;
            mins[i] = mins[i].min(val);
            maxs[i] = maxs[i].max(val);
        }
    }

    // Average the means
    for mean in &mut means {
        *mean /= num_samples as f32;
    }

    // Display or save predictions
    if let Some(output_path) = &args.output {
        save_predictions_with_stats(output_path, &means, &mins, &maxs, &args.format)?;
        info!("Predictions saved to {:?}", output_path);
    }

    if !args.quiet {
        // Display header information
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║         Sundial Forecast Predictions                     ║");
        println!("╚══════════════════════════════════════════════════════════╝");
        println!("\nConfiguration:");
        println!("  Input window size:  {}", window_size);
        println!("  Forecast horizon:   {}", args.horizon);
        if num_samples > 1 {
            println!("  Number of samples:  {}", num_samples);
        }

        // Display predictions in table format
        println!("\n┌─────────┬────────────────┬────────────────┬────────────────┐");
        println!("│ Step    │ Prediction     │ Lower Bound    │ Upper Bound    │");
        println!("├─────────┼────────────────┼────────────────┼────────────────┤");

        for i in 0..forecast_length {
            if num_samples > 1 {
                println!(
                    "│ {:<7} │ {:>14.4} │ {:>14.4} │ {:>14.4} │",
                    i, means[i], mins[i], maxs[i]
                );
            } else {
                println!(
                    "│ {:<7} │ {:>14.4} │ {:>14} │ {:>14} │",
                    i, means[i], "N/A", "N/A"
                );
            }
        }

        println!("└─────────┴────────────────┴────────────────┴────────────────┘");

        // Summary statistics
        let overall_mean: f32 = means.iter().sum::<f32>() / means.len() as f32;
        let overall_min = mins.iter().fold(f32::INFINITY, |a, b| a.min(*b));
        let overall_max = maxs.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));

        println!("\n📊 Summary Statistics:");
        println!("  ┌─────────────────────────────────────┐");
        println!("  │ Mean Forecast:   {:>18.4} │", overall_mean);
        println!("  │ Minimum Value:   {:>18.4} │", overall_min);
        println!("  │ Maximum Value:   {:>18.4} │", overall_max);
        if num_samples > 1 {
            let range = overall_max - overall_min;
            println!("  │ Forecast Range:  {:>18.4} │", range);
        }
        println!("  └─────────────────────────────────────┘");

        // Verbose mode: show additional details
        if args.verbose {
            println!("\n🔍 Detailed Information:");
            // Handle both embedded weights and custom model path
            let model_name = args
                .model
                .as_ref()
                .map(|p| {
                    p.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string()
                })
                .unwrap_or_else(|| "embedded weights".to_string());
            println!("  Model: {}", model_name);
            println!("  Input data points: {}", values.len());
            println!("  Input window used: {:?}", input_window.len());

            // Show first few input values
            println!(
                "  Input window (first 5): {:?}",
                &input_window[..input_window_len.min(5)]
            );
            println!("  Input window length: {}", input_window_len);

            // Show prediction distribution
            let mut sorted_preds = means.clone();
            sorted_preds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = sorted_preds[sorted_preds.len() / 2];
            println!("  Median forecast: {:.4}", median);

            // Calculate standard deviation
            let variance: f32 = means
                .iter()
                .map(|x| (x - overall_mean).powi(2))
                .sum::<f32>()
                / means.len() as f32;
            let std_dev = variance.sqrt();
            println!("  Standard deviation: {:.4}", std_dev);
        }
    }

    Ok(())
}

/// Save predictions to file in specified format (json or csv)
fn save_predictions(path: &PathBuf, predictions: &[f32], format: &str) -> Result<()> {
    let format = format.to_lowercase();

    match format.as_str() {
        "json" => save_predictions_json(path, predictions, &[], &[]),
        "csv" => save_predictions_csv(path, predictions, &[], &[]),
        other => anyhow::bail!("Unsupported output format: {}. Supported: json, csv", other),
    }
}

/// Save predictions with statistics to file in specified format (json or csv)
fn save_predictions_with_stats(
    path: &PathBuf,
    means: &[f32],
    mins: &[f32],
    maxs: &[f32],
    format: &str,
) -> Result<()> {
    let format = format.to_lowercase();

    match format.as_str() {
        "json" => save_predictions_json(path, means, mins, maxs),
        "csv" => save_predictions_csv(path, means, mins, maxs),
        other => anyhow::bail!("Unsupported output format: {}. Supported: json, csv", other),
    }
}

/// Save predictions to JSON format
fn save_predictions_json(
    path: &PathBuf,
    predictions: &[f32],
    mins: &[f32],
    maxs: &[f32],
) -> Result<()> {
    use serde::Serialize;
    use std::fs::File;
    use std::io::BufWriter;

    #[derive(Serialize)]
    struct Prediction {
        timestamp: String,
        predicted_value: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        confidence_lower: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        confidence_upper: Option<f32>,
    }

    // Serialize custom to handle f32 properly
    let predictions_vec: Vec<Prediction> = predictions
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            let ci_lower = if !mins.is_empty() && i < mins.len() {
                Some(mins[i])
            } else {
                None
            };
            let ci_upper = if !maxs.is_empty() && i < maxs.len() {
                Some(maxs[i])
            } else {
                None
            };
            Prediction {
                timestamp: format!("+{}", i),
                predicted_value: val,
                confidence_lower: ci_lower,
                confidence_upper: ci_upper,
            }
        })
        .collect();

    let file = File::create(path).context("Failed to create output file")?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &predictions_vec).context("Failed to write JSON")?;

    Ok(())
}

/// Save predictions to CSV format
fn save_predictions_csv(
    path: &PathBuf,
    predictions: &[f32],
    mins: &[f32],
    maxs: &[f32],
) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;

    // Write header with confidence interval columns if available
    if !mins.is_empty() && !maxs.is_empty() {
        wtr.write_record([
            "timestamp",
            "predicted_value",
            "confidence_lower",
            "confidence_upper",
        ])?;

        // Write data rows with confidence intervals
        for (i, pred) in predictions.iter().enumerate() {
            let ci_lower = if i < mins.len() {
                mins[i].to_string()
            } else {
                "".to_string()
            };
            let ci_upper = if i < maxs.len() {
                maxs[i].to_string()
            } else {
                "".to_string()
            };
            wtr.write_record([&format!("+{}", i), &pred.to_string(), &ci_lower, &ci_upper])?;
        }
    } else {
        wtr.write_record(["timestamp", "predicted_value"])?;

        // Write data rows
        for (i, pred) in predictions.iter().enumerate() {
            wtr.write_record([&format!("+{}", i), &pred.to_string()])?;
        }
    }

    wtr.flush()?;
    Ok(())
}

/// Detect column names from available columns using priority-based matching
/// Returns (timestamp_col, value_col) - either detected or original defaults
fn detect_column_names(
    provided_timestamp_col: &str,
    provided_value_col: &str,
) -> Result<(String, String)> {
    // Only auto-detect if using default column names
    let is_default_timestamp = provided_timestamp_col == "timestamp";
    let is_default_value = provided_value_col == "value";

    if !is_default_timestamp && !is_default_value {
        // User explicitly specified both, no auto-detection needed
        return Ok((
            provided_timestamp_col.to_string(),
            provided_value_col.to_string(),
        ));
    }

    // Get available columns from the input file
    // We need to read the file headers to check for matches
    // This is a bit tricky because we don't have the file path here
    // We'll handle this in the load functions instead
    // For now, return defaults and let load functions handle detection
    Ok((
        provided_timestamp_col.to_string(),
        provided_value_col.to_string(),
    ))
}

/// Auto-detect column names from a list of available column names
/// Returns (detected_timestamp, detected_value) or None if no matches found
fn auto_detect_columns(columns: &[String]) -> Option<(String, String)> {
    // Timestamp candidates in priority order
    let timestamp_candidates = ["time", "datetime", "date", "timestamp", "t"];
    // Value candidates in priority order
    let value_candidates = ["value", "val", "price", "amount", "y", "target"];

    let mut detected_timestamp: Option<String> = None;
    let mut detected_value: Option<String> = None;

    // Detect timestamp column (case-insensitive)
    for &candidate in &timestamp_candidates {
        if let Some(found) = columns.iter().find(|col| col.to_lowercase() == candidate) {
            detected_timestamp = Some(found.clone());
            break; // Take first match in priority order
        }
    }

    // Detect value column (case-insensitive)
    for &candidate in &value_candidates {
        if let Some(found) = columns.iter().find(|col| col.to_lowercase() == candidate) {
            detected_value = Some(found.clone());
            break; // Take first match in priority order
        }
    }

    // Only return if both detected
    match (detected_timestamp, detected_value) {
        (Some(ts), Some(val)) => Some((ts, val)),
        _ => None,
    }
}

/// Display quick-start tutorial for first-time users
fn show_quickstart_guide() {
    println!(
        r#"
╔══════════════════════════════════════════════════════════╗
║     🚀 Sundial Quick-Start Guide                         ║
║     Your 5-minute introduction to time series forecasting ║
╚══════════════════════════════════════════════════════════╝

Welcome to Sundial! This guide will walk you through making your first forecast.

═══════════════════════════════════════════════════════════
📝 STEP 1: Prepare Your Data
═══════════════════════════════════════════════════════════

Create a CSV file with time series data. Required columns:
  • timestamp - Any format (e.g., "2024-01-01", "2024-01-02")
  • value     - Numeric values (e.g., 100.5, 200.0)

Example (save as data.csv):
  timestamp,value
  2024-01-01,100.0
  2024-01-02,105.5
  2024-01-03,103.2
  2024-01-04,108.7
  ...

═══════════════════════════════════════════════════════════
🔍 STEP 2: Preview Your Data
═══════════════════════════════════════════════════════════

Command:
  sundial --input data.csv

Expected output:
  Successfully loaded time series data from: data.csv
  Row count: 100
  Column names: ["timestamp", "value"]

  First 5 rows:
  shape: (5, 2)
  ┌────────────┬───────┐
  │ timestamp  ┆ value │
  │ ---        ┆ ---   │
  │ str        ┆ f64   │
  ╞════════════╪═══════╡
  │ 2024-01-01 ┆ 100.0 │
  │ 2024-01-02 ┆ 105.5 │
  │ ...

═══════════════════════════════════════════════════════════
🎯 STEP 3: Run Your First Forecast
═══════════════════════════════════════════════════════════

Command:
  sundial --infer \
    --model weights/model.safetensors \
    --input data.csv \
    --horizon 10

Expected output:
  ╔══════════════════════════════════════════════════════════╗
  ║         Sundial Forecast Predictions                     ║
  ╚══════════════════════════════════════════════════════════╝

  Configuration:
    Input window size:  30
    Forecast horizon:   10

  ┌─────────┬────────────────┬────────────────┬────────────────┐
  │ Step    │ Prediction     │ Lower Bound    │ Upper Bound    │
  ├─────────┼────────────────┼────────────────┼────────────────┤
  │ 0       │       112.3456 │            N/A │            N/A │
  │ 1       │       113.7821 │            N/A │            N/A │
  │ ...

═══════════════════════════════════════════════════════════
⚙️ STEP 4: Customize Your Forecast
═══════════════════════════════════════════════════════════

Change the forecast horizon (default is 10):
  sundial --infer --model weights/model.safetensors \
    --input data.csv --horizon 24

Use more historical data (default window is 30 points):
  sundial --infer --model weights/model.safetensors \
    --input data.csv --window-size 60

═══════════════════════════════════════════════════════════
🔮 STEP 5: Add Uncertainty Estimation
═══════════════════════════════════════════════════════════

Generate predictions with confidence intervals:
  sundial --infer --model weights/model.safetensors \
    --input data.csv --horizon 10 --num-samples 100

This runs 100 stochastic samples and shows:
  • Prediction (mean)
  • Lower bound (minimum across samples)
  • Upper bound (maximum across samples)

═══════════════════════════════════════════════════════════
💾 STEP 6: Save Your Forecasts
═══════════════════════════════════════════════════════════

Save to JSON:
  sundial --infer --model weights/model.safetensors \
    --input data.csv --output forecasts.json --format json

Save to CSV:
  sundial --infer --model weights/model.safetensors \
    --input data.csv --output forecasts.csv --format csv

═══════════════════════════════════════════════════════════
📚 Next Steps
═══════════════════════════════════════════════════════════

• Run 'sundial --help' for complete documentation
• Explore advanced options like custom configurations
• Check the project README for model training details

Happy forecasting! 🎉
"#
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_csv_validation() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("test.csv");

        // Create valid CSV
        let mut file = File::create(&csv_path).unwrap();
        writeln!(file, "timestamp,value").unwrap();
        writeln!(file, "2024-01-01,100.0").unwrap();
        writeln!(file, "2024-01-02,101.0").unwrap();

        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(load_and_display_data(&csv_path, "timestamp", "value", true));

        assert!(result.is_ok());
    }

    #[test]
    fn test_missing_file() {
        let path = PathBuf::from("/nonexistent/file.csv");
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(load_and_display_data(&path, "timestamp", "value", true));

        assert!(result.is_err());
    }

    #[test]
    fn test_missing_columns() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("bad.csv");

        // Create CSV without required columns (using names that won't be auto-detected)
        let mut file = File::create(&csv_path).unwrap();
        writeln!(file, "datetime_col,value_col").unwrap();
        writeln!(file, "2024-01-01,100.0").unwrap();

        // When using defaults, auto-detection should fail and fall back to defaults
        // which then fail because the columns don't exist
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(load_and_display_data(&csv_path, "timestamp", "value", true));

        assert!(result.is_err());
    }

    #[test]
    fn test_parquet_validation() {
        let temp_dir = TempDir::new().unwrap();
        let parquet_path = temp_dir.path().join("test.parquet");

        // Create a valid parquet file using polars
        use polars::prelude::*;

        let timestamps = Series::new(
            "timestamp".into(),
            ["2024-01-01", "2024-01-02", "2024-01-03"],
        );
        let values = Series::new("value".into(), [100.0, 101.0, 102.0]);
        let df = DataFrame::new(vec![Column::from(timestamps), Column::from(values)]).unwrap();

        // Write to parquet
        let file = File::create(&parquet_path).unwrap();
        ParquetWriter::new(file).finish(&mut df.clone()).unwrap();

        // Test loading
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(load_and_display_data(
                &parquet_path,
                "timestamp",
                "value",
                true,
            ));

        assert!(result.is_ok());
    }

    #[test]
    fn test_parquet_missing_columns() {
        let temp_dir = TempDir::new().unwrap();
        let parquet_path = temp_dir.path().join("bad.parquet");

        use polars::prelude::*;

        // Create parquet without required columns (using names that won't be auto-detected)
        let datetime_cols = Series::new("datetime_col".into(), ["2024-01-01", "2024-01-02"]);
        let value_cols = Series::new("value_col".into(), [100.0, 101.0]);
        let df =
            DataFrame::new(vec![Column::from(datetime_cols), Column::from(value_cols)]).unwrap();

        let file = File::create(&parquet_path).unwrap();
        ParquetWriter::new(file).finish(&mut df.clone()).unwrap();

        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(load_and_display_data(
                &parquet_path,
                "timestamp",
                "value",
                true,
            ));

        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_format() {
        let temp_dir = TempDir::new().unwrap();
        let txt_path = temp_dir.path().join("test.txt");

        let mut file = File::create(&txt_path).unwrap();
        writeln!(file, "some text").unwrap();

        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(load_and_display_data(&txt_path, "timestamp", "value", true));

        assert!(result.is_err());
    }

    #[test]
    fn test_custom_column_names() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("custom.csv");

        // Create CSV with custom column names
        let mut file = File::create(&csv_path).unwrap();
        writeln!(file, "time,price").unwrap();
        writeln!(file, "2024-01-01,100.0").unwrap();
        writeln!(file, "2024-01-02,101.0").unwrap();

        // Should succeed with custom column names
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(load_and_display_data(&csv_path, "time", "price", true));

        assert!(result.is_ok());
    }

    #[test]
    fn test_custom_column_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("custom.csv");

        // Create CSV with custom column names
        let mut file = File::create(&csv_path).unwrap();
        writeln!(file, "time,price").unwrap();
        writeln!(file, "2024-01-01,100.0").unwrap();

        // Should fail when looking for non-existent column
        // Note: auto-detection will find "time" and "price" when using defaults,
        // so we explicitly request non-existent columns
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(load_and_display_data(
                &csv_path,
                "nonexistent_ts",
                "nonexistent_val",
                true,
            ));

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("not found"));
    }

    #[test]
    fn test_auto_detect_timestamp_columns() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("auto_detect.csv");

        // Create CSV with auto-detectable column names
        let mut file = File::create(&csv_path).unwrap();
        writeln!(file, "time,price").unwrap();
        writeln!(file, "2024-01-01,100.0").unwrap();
        writeln!(file, "2024-01-02,101.0").unwrap();

        // Should auto-detect "time" and "price" columns
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(load_and_display_data(&csv_path, "timestamp", "value", true));

        assert!(result.is_ok());
    }

    #[test]
    fn test_auto_detect_priority_order() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("priority.csv");

        // Create CSV with multiple timestamp candidates - should pick highest priority
        let mut file = File::create(&csv_path).unwrap();
        writeln!(file, "time,datetime,date,value").unwrap();
        writeln!(file, "2024-01-01,2024-01-01,2024-01-01,100.0").unwrap();

        // Should auto-detect "time" (highest priority) and "value"
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(load_and_display_data(&csv_path, "timestamp", "value", true));

        assert!(result.is_ok());
    }

    #[test]
    fn test_inference_with_custom_columns() {
        // Test that flexible column names work with inference pipeline
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("inference_test.csv");

        // Create CSV with custom column names
        let mut file = File::create(&csv_path).unwrap();
        writeln!(file, "time,price").unwrap();
        for i in 0..50 {
            writeln!(
                file,
                "2024-01-{:02},{}.{:02}",
                i + 1,
                100 + i,
                (i * 7) % 100
            )
            .unwrap();
        }

        // Load data with custom column names
        let result = load_csv(&csv_path, "time", "price");

        assert!(result.is_ok(), "Should load data with custom column names");

        let df = result.unwrap();
        assert_eq!(df.height(), 50, "Should have 50 rows");

        // Verify the correct columns were used
        let columns: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(columns.contains(&"time".to_string()));
        assert!(columns.contains(&"price".to_string()));
    }

    #[test]
    fn test_inference_with_auto_detected_columns() {
        // Test that auto-detection works for inference pipeline
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("auto_inference.csv");

        // Create CSV with auto-detectable column names
        let mut file = File::create(&csv_path).unwrap();
        writeln!(file, "datetime,amount").unwrap();
        for i in 0..50 {
            writeln!(
                file,
                "2024-01-{:02},{}.{:02}",
                i + 1,
                100 + i,
                (i * 7) % 100
            )
            .unwrap();
        }

        // Load data with defaults - should auto-detect "datetime" and "amount"
        let result = load_csv(&csv_path, "timestamp", "value");

        assert!(result.is_ok(), "Should auto-detect columns and load data");

        let df = result.unwrap();
        assert_eq!(df.height(), 50, "Should have 50 rows");
    }

    #[test]
    fn test_inference_params_validation() {
        // Test that inference parameters are validated correctly
        let mut args = Args {
            input: Some(PathBuf::from("test.csv")),
            output: Some(PathBuf::from("output.json")),
            horizon: 0, // Invalid: must be positive
            window_size: 30,
            num_samples: 1,
            model: Some(PathBuf::from("model.safetensors")),
            config: None,
            format: "json".to_string(),
            quiet: false,
            verbose: false,
            infer: true,
            quickstart: false,
            timestamp_col: "timestamp".to_string(),
            value_col: "value".to_string(),
            no_auto_detect: false,
        };

        let result = validate_forecast_params(&args);
        assert!(result.is_err(), "Should fail with horizon=0");
        assert!(result.unwrap_err().to_string().contains("horizon"));

        // Test with valid params
        args.horizon = 10;
        let result = validate_forecast_params(&args);
        assert!(result.is_ok(), "Should pass with valid horizon");

        // Test with invalid window_size
        args.window_size = 0;
        let result = validate_forecast_params(&args);
        assert!(result.is_err(), "Should fail with window_size=0");

        // Test with invalid num_samples
        args.window_size = 30;
        args.num_samples = 0;
        let result = validate_forecast_params(&args);
        assert!(result.is_err(), "Should fail with num_samples=0");
    }

    #[test]
    fn test_model_flag_nonexistent_path() {
        // Test that --model flag with non-existent path returns clear error
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("test.csv");

        // Create a valid CSV
        let mut file = File::create(&csv_path).unwrap();
        writeln!(file, "timestamp,value").unwrap();
        writeln!(file, "2024-01-01,100.0").unwrap();
        writeln!(file, "2024-01-02,101.0").unwrap();

        let args = Args {
            input: Some(csv_path),
            output: None,
            horizon: 10,
            window_size: 30,
            num_samples: 1,
            model: Some(PathBuf::from("/nonexistent/path/model.safetensors")),
            config: None,
            format: "json".to_string(),
            quiet: true,
            verbose: false,
            infer: true,
            quickstart: false,
            timestamp_col: "timestamp".to_string(),
            value_col: "value".to_string(),
            no_auto_detect: false,
        };

        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(run_inference(&args, "timestamp", "value"));

        assert!(result.is_err(), "Should fail with non-existent model path");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Model file not found"),
            "Error should mention 'Model file not found'"
        );
        assert!(
            err_msg.contains("--model"),
            "Error should mention --model flag"
        );
    }

    #[test]
    fn test_model_flag_directory_instead_of_file() {
        // Test that --model flag with directory path returns clear error
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("test.csv");

        // Create a valid CSV
        let mut file = File::create(&csv_path).unwrap();
        writeln!(file, "timestamp,value").unwrap();
        writeln!(file, "2024-01-01,100.0").unwrap();

        let args = Args {
            input: Some(csv_path),
            output: None,
            horizon: 10,
            window_size: 30,
            num_samples: 1,
            model: Some(temp_dir.path().to_path_buf()), // Pass directory, not file
            config: None,
            format: "json".to_string(),
            quiet: true,
            verbose: false,
            infer: true,
            quickstart: false,
            timestamp_col: "timestamp".to_string(),
            value_col: "value".to_string(),
            no_auto_detect: false,
        };

        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(run_inference(&args, "timestamp", "value"));

        assert!(result.is_err(), "Should fail with directory path");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not a file"),
            "Error should mention 'not a file'"
        );
    }

    #[cfg(feature = "integration-tests")]
    #[test]
    fn test_model_flag_integration() {
        // Integration test: Verify --model flag works with actual model file
        // This test requires the weights/model.safetensors file to exist
        #[cfg(not(feature = "integration-tests"))]
        panic!("Integration tests require --features integration-tests");

        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("test.csv");

        // Create test data
        let mut file = File::create(&csv_path).unwrap();
        writeln!(file, "timestamp,value").unwrap();
        for i in 0..50 {
            writeln!(
                file,
                "2024-01-{:02},{}.{:02}",
                i + 1,
                100.0 + i as f32 * 0.5,
                (i * 7) % 100
            )
            .unwrap();
        }

        let model_path = PathBuf::from("weights/model.safetensors");
        if !model_path.exists() {
            println!("Skipping integration test: weights/model.safetensors not found");
            return;
        }

        let args = Args {
            input: Some(csv_path),
            output: None,
            horizon: 5,
            window_size: 30,
            num_samples: 1,
            model: Some(model_path),
            config: None,
            format: "json".to_string(),
            quiet: true,
            verbose: false,
            infer: true,
            quickstart: false,
            timestamp_col: "timestamp".to_string(),
            value_col: "value".to_string(),
            no_auto_detect: false,
        };

        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(run_inference(&args, "timestamp", "value"));

        assert!(result.is_ok(), "Should succeed with valid model path");
    }
}
