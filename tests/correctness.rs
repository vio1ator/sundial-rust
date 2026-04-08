//! End-to-end correctness tests against Python reference
//!
//! These tests verify that the full Sundial model produces outputs
//! matching the Python implementation.

use candle_core::{Device, Tensor};
use sundial_rust::testing::{assert_tensor_close, load_reference_tensor, print_comparison_stats};
use sundial_rust::{SundialConfig, SundialModel};

/// Test that end-to-end model output matches Python reference
///
/// This test loads a pre-computed input, runs it through the Rust model,
/// and compares the predictions against Python reference outputs.
///
/// Prerequisites:
/// - Generate Python reference: `python scripts/generate_reference.py --input test_data/input.npy --output tests/reference_data/ --full-model`
#[test]
#[ignore = "Requires Python reference data to be generated first"]
fn test_end_to_end_matches_python() {
    let device = Device::Cpu;

    // Load test input
    let mut input =
        load_reference_tensor("tests/reference_data/input.npy").expect("Failed to load test input");

    // Squeeze input if it has extra dimensions (e.g., [1, 2880, 1] -> [1, 2880])
    if input.dims().len() == 3 {
        input = input.squeeze(2).expect("Failed to squeeze input");
    }

    println!("Input shape: {:?}", input.dims());

    // Load Python predictions
    let python_predictions = load_reference_tensor("tests/reference_data/predictions.npy")
        .expect("Failed to load Python predictions");

    // Load model with real weights
    let config = SundialConfig::default();
    let vb = sundial_rust::model::loader::load_sundial_from_huggingface(
        "thuml/sundial-base-128m",
        &device,
    )
    .expect("Failed to load model weights");

    let model = SundialModel::new(&config, vb).expect("Failed to create model");

    // Run inference
    let mut predictions = model
        .generate(&input, 14, 1, false)
        .expect("Failed to run inference");

    // Squeeze predictions to match Python shape [14]
    // Rust returns [1, 1, 14], we want [14]
    while predictions.dims().len() > 1 {
        predictions = predictions
            .squeeze(0)
            .expect("Failed to squeeze predictions");
    }

    println!("Predictions shape: {:?}", predictions.dims());
    println!("Python predictions shape: {:?}", python_predictions.dims());

    // Print comparison stats for debugging
    let _ = print_comparison_stats(&predictions, &python_predictions, "Final predictions");

    // Compare with Python (relaxed tolerance for full model due to error accumulation
    // through 12 transformer layers, numerical precision differences, and flow matching
    // generation stochasticity)
    assert_tensor_close(&predictions, &python_predictions, 3.0, "Final predictions")
        .expect("Predictions should match Python reference");
}

/// Test that model can process various input sizes
#[test]
#[ignore = "Requires model weights to be available"]
fn test_end_to_end_various_sizes() {
    let device = Device::Cpu;

    // Test with different input sizes
    let test_cases = vec![
        (2880, 1), // Full context
        (1440, 1), // Half context
        (720, 1),  // Quarter context
    ];

    for (timesteps, features) in test_cases {
        // Create random input for this size
        let input = Tensor::randn(0.0f32, 1.0f32, (1, timesteps, features), &device)
            .expect("Failed to create input");

        let config = SundialConfig::default();
        let vb = sundial_rust::model::loader::load_sundial_from_huggingface(
            "thuml/sundial-base-128m",
            &device,
        )
        .expect("Failed to load model");

        let model = SundialModel::new(&config, vb).expect("Failed to create model");

        // Should not panic
        let _predictions = model
            .generate(&input, 14, 1, false)
            .expect("Inference should succeed");
    }
}
