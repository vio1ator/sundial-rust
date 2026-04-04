//! Compare intermediate tensors between Rust and Python implementations
//!
//! This example runs the Sundial model in Rust with debug mode enabled,
//! saving intermediate tensors that can be compared with Python outputs.
//!
//! Usage:
//!     SUNDIAL_DEBUG=1 cargo run --release --example compare_intermediates
//!
//! Then compare with Python outputs using:
//!     python scripts/compare_tensors.py

use candle_core::{Device, Tensor};
use sundial_rust::{debug_utils, SundialConfig, SundialModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Sundial Intermediate Tensor Comparison");
    println!("======================================\n");

    // Check if debug mode is enabled
    let debug_mode = std::env::var("SUNDIAL_DEBUG").is_ok();
    if !debug_mode {
        println!("Warning: SUNDIAL_DEBUG environment variable not set.");
        println!("Set it to enable debug output: SUNDIAL_DEBUG=1 cargo run --release --example compare_intermediates\n");
    }

    // Create config
    let config = SundialConfig::sundial_base_128m();
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

    // Create a random input (matching Python)
    let batch_size = 1;
    let seq_len = 2880;
    println!("Creating random input: shape=[{}, {}]", batch_size, seq_len);
    let input = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len), &device)?;

    debug_utils::debug_tensor("input", &input);

    // Save input for comparison
    debug_utils::save_tensor_to_bin("input", &input)?;

    // Note: We can't load actual weights without the safetensors file,
    // so we'll just demonstrate the debug infrastructure
    println!("\nNote: This example requires actual model weights to run full inference.");
    println!("To test with real weights, use the forecast example with debug enabled.\n");

    // Create a small test model for demonstration
    println!("Creating small test model for demonstration...");
    let test_config = SundialConfig {
        input_token_len: 16,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        output_token_lens: vec![32],
        ..Default::default()
    };

    let vb = candle_nn::VarBuilder::from_varmap(
        &candle_nn::VarMap::new(),
        candle_core::DType::F32,
        &device,
    );

    let model = SundialModel::new(&test_config, vb)?;

    // Create smaller input for test model
    let test_input = Tensor::randn(0.0f32, 1.0f32, (1, 64), &device)?;

    println!("\nRunning inference with test model...");
    let output = model.generate(&test_input, 32, 3, false)?;

    println!("\nOutput shape: {:?}", output.dims());
    debug_utils::debug_tensor("final_output", &output);

    // Save output for comparison
    debug_utils::save_tensor_to_bin("final_output", &output)?;

    println!("\nDebug outputs saved to /tmp/*.bin");
    println!("Use python scripts/compare_tensors.py to compare with Python outputs");

    Ok(())
}
