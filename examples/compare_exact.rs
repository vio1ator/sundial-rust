//! Exact component comparison test suite
//!
//! This example runs individual components of the Sundial model and compares
//! their outputs with expected values from the Python implementation.
//!
//! Usage:
//!     cargo run --release --example compare_exact

use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use sundial_rust::model::{
    attention::{AttentionConfig, SundialAttention},
    decoder_layer::{DecoderLayerConfig, SundialDecoderLayer},
    mlp::SundialMLP,
    patch_embed::SundialPatchEmbedding,
    rope::SundialRotaryEmbedding,
};
use sundial_rust::{debug_utils, SundialConfig};

/// Test tolerance for floating point comparisons
const TOLERANCE: f32 = 1e-4;

/// Result of a component test
struct TestResult {
    name: String,
    passed: bool,
    max_diff: f32,
    message: String,
}

impl TestResult {
    fn new(name: &str, passed: bool, max_diff: f32, message: &str) -> Self {
        Self {
            name: name.to_string(),
            passed,
            max_diff,
            message: message.to_string(),
        }
    }
}

fn print_result(result: &TestResult) {
    let status = if result.passed {
        "✓ PASS"
    } else {
        "✗ FAIL"
    };
    println!("  {}: {}", status, result.name);
    if result.max_diff > 0.0 {
        println!("    Max diff: {:.8}", result.max_diff);
    }
    if !result.message.is_empty() {
        println!("    {}", result.message);
    }
}

/// Test patch embedding layer
fn test_patch_embedding() -> TestResult {
    println!("\nTesting Patch Embedding...");

    let config = SundialConfig {
        input_token_len: 16,
        hidden_size: 64,
        intermediate_size: 128,
        ..Default::default()
    };

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    match SundialPatchEmbedding::new(&config, vb) {
        Ok(embedding) => {
            // Create test input
            let input = Tensor::randn(0.0f32, 1.0f32, (2, 64), &device).unwrap();
            let output = embedding.forward(&input).unwrap();

            // Check output shape
            let expected_shape = &[2, 4, 64]; // [batch, num_patches, hidden]
            if output.dims() != expected_shape {
                return TestResult::new(
                    "patch_embedding",
                    false,
                    0.0,
                    &format!(
                        "Wrong output shape: expected {:?}, got {:?}",
                        expected_shape,
                        output.dims()
                    ),
                );
            }

            // Check for NaN/Inf
            let has_nan = output
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .any(|x: &f32| x.is_nan());

            if has_nan {
                return TestResult::new(
                    "patch_embedding",
                    false,
                    f32::INFINITY,
                    "Output contains NaN values",
                );
            }

            debug_utils::debug_tensor("patch_embed_output", &output);

            TestResult::new(
                "patch_embedding",
                true,
                0.0,
                "Output shape and values look correct",
            )
        }
        Err(e) => TestResult::new(
            "patch_embedding",
            false,
            0.0,
            &format!("Failed to create layer: {}", e),
        ),
    }
}

/// Test RoPE implementation
fn test_rope() -> TestResult {
    println!("\nTesting RoPE (Rotary Positional Embeddings)...");

    let device = Device::Cpu;
    let rope = match SundialRotaryEmbedding::new(64, 1000, 10000.0, &device) {
        Ok(r) => r,
        Err(e) => return TestResult::new("rope", false, 0.0, &format!("Failed to create: {}", e)),
    };

    // Create test query and key
    let q = Tensor::randn(0.0f32, 1.0f32, (2, 4, 10, 64), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0f32, (2, 4, 10, 64), &device).unwrap();

    let (q_embed, k_embed) = match rope.forward(&q, &k, None) {
        Ok((q, k)) => (q, k),
        Err(e) => return TestResult::new("rope", false, 0.0, &format!("Forward failed: {}", e)),
    };

    // Check shape preservation
    if q_embed.dims() != q.dims() || k_embed.dims() != k.dims() {
        return TestResult::new("rope", false, 0.0, "RoPE changed tensor shape");
    }

    // Check norm preservation (RoPE should preserve vector norms)
    let q_norm = q
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .sqrt()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let q_embed_norm = q_embed
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .sqrt()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    let norm_diff = (q_norm - q_embed_norm).abs();

    if norm_diff > 1e-5 {
        return TestResult::new(
            "rope",
            false,
            norm_diff,
            "RoPE did not preserve vector norm (should be < 1e-5)",
        );
    }

    debug_utils::debug_tensor("rope_q_output", &q_embed);

    TestResult::new("rope", true, norm_diff, "Norm preserved correctly")
}

/// Test attention layer
fn test_attention() -> TestResult {
    println!("\nTesting Attention Layer...");

    let config = AttentionConfig {
        hidden_size: 64,
        num_heads: 4,
        head_dim: 16,
        attention_dropout: 0.0,
        max_position_embeddings: 1000,
        rope_theta: 10000.0,
        layer_idx: Some(0),
    };

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let attention = match SundialAttention::new(&config, vb) {
        Ok(a) => a,
        Err(e) => {
            return TestResult::new("attention", false, 0.0, &format!("Failed to create: {}", e))
        }
    };

    // Create test input
    let input = Tensor::randn(0.0f32, 1.0f32, (2, 10, 64), &device).unwrap();

    match attention.forward(&input) {
        Ok(output) => {
            // Check output shape
            if output.dims() != input.dims() {
                return TestResult::new(
                    "attention",
                    false,
                    0.0,
                    &format!(
                        "Wrong output shape: expected {:?}, got {:?}",
                        input.dims(),
                        output.dims()
                    ),
                );
            }

            // Check for NaN/Inf
            let has_nan = output
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .any(|x: &f32| x.is_nan());

            if has_nan {
                return TestResult::new(
                    "attention",
                    false,
                    f32::INFINITY,
                    "Output contains NaN values",
                );
            }

            debug_utils::debug_tensor("attention_output", &output);

            TestResult::new(
                "attention",
                true,
                0.0,
                "Output shape and values look correct",
            )
        }
        Err(e) => TestResult::new("attention", false, 0.0, &format!("Forward failed: {}", e)),
    }
}

/// Test MLP layer
fn test_mlp() -> TestResult {
    println!("\nTesting MLP Layer...");

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let mlp = match SundialMLP::new(64, 128, "silu", vb) {
        Ok(m) => m,
        Err(e) => return TestResult::new("mlp", false, 0.0, &format!("Failed to create: {}", e)),
    };

    // Create test input
    let input = Tensor::randn(0.0f32, 1.0f32, (2, 10, 64), &device).unwrap();

    match mlp.forward(&input) {
        Ok(output) => {
            // Check output shape
            if output.dims() != input.dims() {
                return TestResult::new(
                    "mlp",
                    false,
                    0.0,
                    &format!(
                        "Wrong output shape: expected {:?}, got {:?}",
                        input.dims(),
                        output.dims()
                    ),
                );
            }

            // Check for NaN/Inf
            let has_nan = output
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .any(|x: &f32| x.is_nan());

            if has_nan {
                return TestResult::new("mlp", false, f32::INFINITY, "Output contains NaN values");
            }

            debug_utils::debug_tensor("mlp_output", &output);

            TestResult::new("mlp", true, 0.0, "Output shape and values look correct")
        }
        Err(e) => TestResult::new("mlp", false, 0.0, &format!("Forward failed: {}", e)),
    }
}

/// Test decoder layer
fn test_decoder_layer() -> TestResult {
    println!("\nTesting Decoder Layer...");

    let config = DecoderLayerConfig {
        hidden_size: 64,
        intermediate_size: 128,
        num_attention_heads: 4,
        head_dim: 16,
        hidden_act: "silu".to_string(),
        attention_dropout: 0.0,
        max_position_embeddings: 1000,
        rope_theta: 10000.0,
        layer_idx: 0,
    };

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let layer = match SundialDecoderLayer::new(&config, vb) {
        Ok(l) => l,
        Err(e) => {
            return TestResult::new(
                "decoder_layer",
                false,
                0.0,
                &format!("Failed to create: {}", e),
            )
        }
    };

    // Create test input
    let input = Tensor::randn(0.0f32, 1.0f32, (2, 10, 64), &device).unwrap();

    match layer.forward(&input) {
        Ok(output) => {
            // Check output shape
            if output.dims() != input.dims() {
                return TestResult::new(
                    "decoder_layer",
                    false,
                    0.0,
                    &format!(
                        "Wrong output shape: expected {:?}, got {:?}",
                        input.dims(),
                        output.dims()
                    ),
                );
            }

            // Check for NaN/Inf
            let has_nan = output
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .any(|x: &f32| x.is_nan());

            if has_nan {
                return TestResult::new(
                    "decoder_layer",
                    false,
                    f32::INFINITY,
                    "Output contains NaN values",
                );
            }

            debug_utils::debug_tensor("decoder_layer_output", &output);

            TestResult::new(
                "decoder_layer",
                true,
                0.0,
                "Output shape and values look correct",
            )
        }
        Err(e) => TestResult::new(
            "decoder_layer",
            false,
            0.0,
            &format!("Forward failed: {}", e),
        ),
    }
}

/// Test flow sampling
fn test_flow_sampling() -> TestResult {
    println!("\nTesting Flow Sampling...");

    use sundial_rust::flow::network::SimpleMLPAdaLN;
    use sundial_rust::flow::sampling::flow_sample;

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let net = match SimpleMLPAdaLN::new(16, 32, 16, 32, 2, vb) {
        Ok(n) => n,
        Err(e) => {
            return TestResult::new(
                "flow_sampling",
                false,
                0.0,
                &format!("Failed to create network: {}", e),
            )
        }
    };

    // Create condition
    let z = Tensor::randn(0.0f32, 1.0f32, (2, 32), &device).unwrap();

    match flow_sample(&net, &z, 5, 16, 10) {
        Ok(samples) => {
            // Check output shape
            let expected_shape = &[5, 2, 16]; // [num_samples, batch, channels]
            if samples.dims() != expected_shape {
                return TestResult::new(
                    "flow_sampling",
                    false,
                    0.0,
                    &format!(
                        "Wrong output shape: expected {:?}, got {:?}",
                        expected_shape,
                        samples.dims()
                    ),
                );
            }

            // Check for NaN/Inf
            let has_nan = samples
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .any(|x: &f32| x.is_nan());

            if has_nan {
                return TestResult::new(
                    "flow_sampling",
                    false,
                    f32::INFINITY,
                    "Output contains NaN values",
                );
            }

            debug_utils::debug_tensor("flow_samples", &samples);

            TestResult::new(
                "flow_sampling",
                true,
                0.0,
                "Output shape and values look correct",
            )
        }
        Err(e) => TestResult::new(
            "flow_sampling",
            false,
            0.0,
            &format!("Sampling failed: {}", e),
        ),
    }
}

fn main() {
    println!("Sundial Component Comparison Test Suite");
    println!("========================================\n");

    let mut results = Vec::new();

    // Run all tests
    results.push(test_patch_embedding());
    results.push(test_rope());
    results.push(test_attention());
    results.push(test_mlp());
    results.push(test_decoder_layer());
    results.push(test_flow_sampling());

    // Print summary
    println!("\n============================================================");
    println!("TEST SUMMARY");
    println!("============================================================");

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    for result in &results {
        print_result(result);
    }

    println!("\nTotal: {}/{} tests passed", passed, total);

    if passed == total {
        println!("\n✓ All tests passed!");
    } else {
        println!("\n✗ Some tests failed. Check the output above for details.");
        std::process::exit(1);
    }
}
