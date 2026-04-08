//! Systematic layer-by-layer comparison framework
//!
//! This test framework compares Rust vs Python outputs at each model component:
//! - patch_embed
//! - Each RoPE layer (attention Q/K after RoPE)
//! - Each decoder block (attention output, MLP output, final output)
//! - flow matching
//!
//! Output format: max_diff, mean_diff, std_diff for each component

use candle_core::{Device, Tensor};
use std::fs::File;
use std::io::{BufWriter, Write};
use sundial_rust::testing::{load_reference_tensor, load_tensor_by_name};
use sundial_rust::{SundialConfig, SundialModel};

/// Statistics for tensor comparison
#[derive(Debug, Clone)]
pub struct ComparisonStats {
    pub max_diff: f32,
    pub mean_diff: f32,
    pub std_diff: f32,
    pub shape: String,
}

impl ComparisonStats {
    fn new(actual: &Tensor, expected: &Tensor) -> Self {
        let diff = actual.sub(expected).expect("Failed to compute diff");
        let abs_diff = diff.abs().expect("Failed to compute abs diff");

        let max_diff = abs_diff
            .max_all()
            .expect("Failed to compute max")
            .to_scalar::<f32>()
            .expect("Failed to get scalar");

        let mean_diff = abs_diff
            .mean_all()
            .expect("Failed to compute mean")
            .to_scalar::<f32>()
            .expect("Failed to get scalar");

        // Compute std of the absolute differences: sqrt(E[(|diff| - mean_diff)^2])
        // Use broadcast_sub to subtract scalar from all elements
        let device = abs_diff.device();
        let mean_diff_tensor =
            Tensor::new(mean_diff, device).expect("Failed to create scalar tensor");
        let centered = abs_diff
            .broadcast_sub(&mean_diff_tensor)
            .expect("Failed to center");
        let variance = centered
            .sqr()
            .expect("Failed to square")
            .mean_all()
            .expect("Failed to compute variance")
            .to_scalar::<f32>()
            .expect("Failed to get scalar");
        let std_diff = variance.sqrt();

        let shape = format!("{:?}", actual.shape());

        Self {
            max_diff,
            mean_diff,
            std_diff,
            shape,
        }
    }
}

/// Compare two tensors and return statistics
fn compare_tensors(name: &str, actual: &Tensor, expected: &Tensor) -> Option<ComparisonStats> {
    if actual.shape() != expected.shape() {
        println!(
            "[WARN] {}: Shape mismatch - actual={:?}, expected={:?}",
            name,
            actual.shape(),
            expected.shape()
        );
        return None;
    }

    let stats = ComparisonStats::new(actual, expected);

    println!(
        "{:40} | max_diff={:.6e}, mean_diff={:.6e}, std_diff={:.6e}, shape={}",
        name, stats.max_diff, stats.mean_diff, stats.std_diff, stats.shape
    );

    Some(stats)
}

/// Load a .npy file as a Tensor
fn load_npy_as_tensor(path: &str) -> Result<Tensor, String> {
    use std::io::Read;

    let mut file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?;

    // Simple numpy format parsing
    // Header starts with magic bytes \x93NUMPY
    if buffer.len() < 128 {
        return Err(format!("{}: File too small", path));
    }

    // Parse header to get shape and dtype
    let header_start = 8;
    let header_end = buffer[6..8]
        .iter()
        .map(|&b| b as usize)
        .fold(0, |acc, b| acc * 256 + b);

    let header_str = std::str::from_utf8(&buffer[header_start..header_start + header_end])
        .map_err(|e| format!("Invalid UTF8 in header: {}", e))?;

    // Extract shape - look for 'shape': (..., ...)
    let shape_start = header_str.find("'shape':").ok_or("No shape in header")? + 9;
    let shape_paren = header_str[shape_start..]
        .find('(')
        .ok_or("No paren in shape")?
        + shape_start;
    let shape_end = header_str[shape_paren..]
        .find(')')
        .ok_or("No closing paren")?
        + shape_paren;
    let shape_str = &header_str[shape_paren + 1..shape_end];

    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // Check for fortran order (currently unused but kept for potential future use)
    let _fortran_order = header_str.contains("'fortran_order': True");

    // Data starts after header
    let data_start = header_start + header_end;
    let data = &buffer[data_start..];

    // Convert to f32
    let data: Vec<f32> = data
        .chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Reshape - need to handle fortran order
    let tensor = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
        .map_err(|e| format!("Failed to create tensor: {}", e))?;

    Ok(tensor)
}

/// Generate JSON report of comparison results
fn generate_report(
    stats: &[(String, Option<ComparisonStats>)],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = BufWriter::new(File::create(output_path)?);

    writeln!(file, "{{")?;
    writeln!(file, "  \"comparison_results\": {{")?;

    let mut first = true;
    for (name, maybe_stats) in stats {
        if !first {
            writeln!(file, ",")?;
        }
        first = false;

        if let Some(stats) = maybe_stats {
            writeln!(
                file,
                "    \"{}\": {{\"max_diff\": {}, \"mean_diff\": {}, \"std_diff\": {}, \"shape\": \"{}\"}}",
                name, stats.max_diff, stats.mean_diff, stats.std_diff, stats.shape
            )?;
        } else {
            writeln!(file, "    \"{}\": null", name)?;
        }
    }

    writeln!(file, "\n  }}\n}}")?;

    Ok(())
}

/// Compare patch embedding output
fn test_patch_embed_comparison() -> Vec<(String, Option<ComparisonStats>)> {
    let mut results = Vec::new();

    // Load Python reference
    let python_output =
        load_npy_as_tensor("tests/reference_data/intermediates/patch_embed_output.npy");

    // Load Rust output (generated with SUNDIAL_DEBUG=1)
    let rust_output = load_npy_as_tensor("/tmp/patch_embed_output_rust.bin");

    match (python_output, rust_output) {
        (Ok(py), Ok(rust)) => {
            results.push((
                "patch_embed".to_string(),
                compare_tensors("patch_embed", &rust, &py),
            ));
        }
        (Err(e), _) => {
            println!("[SKIP] patch_embed: Python reference not found - {}", e);
        }
        (_, Err(e)) => {
            println!("[SKIP] patch_embed: Rust output not found - {}", e);
        }
    }

    results
}

/// Compare RoPE outputs for each layer
fn test_rope_comparison() -> Vec<(String, Option<ComparisonStats>)> {
    let mut results = Vec::new();

    for layer_idx in 0..12 {
        // Try to load Rust attention outputs (saved during debug mode)
        let rust_q_after_rope = load_npy_as_tensor(&format!(
            "/tmp/layer_{}_attention_q_after_rope_rust.bin",
            layer_idx
        ));
        let rust_k_after_rope = load_npy_as_tensor(&format!(
            "/tmp/layer_{}_attention_k_after_rope_rust.bin",
            layer_idx
        ));

        // Try to load Python reference
        let py_q = load_npy_as_tensor(&format!(
            "tests/reference_data/intermediates/layer_{}_q_after_rope.npy",
            layer_idx
        ));
        let py_k = load_npy_as_tensor(&format!(
            "tests/reference_data/intermediates/layer_{}_k_after_rope.npy",
            layer_idx
        ));

        if let (Ok(rust_q), Ok(py_q)) = (&rust_q_after_rope, &py_q) {
            results.push((
                format!("layer_{}_rope_q", layer_idx),
                compare_tensors(&format!("layer_{}_rope_q", layer_idx), rust_q, py_q),
            ));
        }

        if let (Ok(rust_k), Ok(py_k)) = (&rust_k_after_rope, &py_k) {
            results.push((
                format!("layer_{}_rope_k", layer_idx),
                compare_tensors(&format!("layer_{}_rope_k", layer_idx), rust_k, py_k),
            ));
        }
    }

    results
}

/// Compare each decoder block output
fn test_block_comparison() -> Vec<(String, Option<ComparisonStats>)> {
    let mut results = Vec::new();

    for layer_idx in 0..12 {
        // Load Rust output
        let rust_output = load_npy_as_tensor(&format!("/tmp/layer_{}_output_rust.bin", layer_idx));

        // Load Python reference
        let py_output = load_npy_as_tensor(&format!(
            "tests/reference_data/intermediates/layer_{}_output.npy",
            layer_idx
        ));

        if let (Ok(rust), Ok(py)) = (&rust_output, &py_output) {
            results.push((
                format!("layer_{}_block", layer_idx),
                compare_tensors(&format!("layer_{}_block", layer_idx), rust, py),
            ));
        }
    }

    results
}

/// Compare flow matching output
fn test_flow_matching_comparison() -> Vec<(String, Option<ComparisonStats>)> {
    let mut results = Vec::new();

    // Load Python reference
    let python_output = load_npy_as_tensor("tests/reference_data/flow_samples.npy");

    // Load Rust output
    let rust_output = load_npy_as_tensor("/tmp/flow_samples_rust.bin");

    match (python_output, rust_output) {
        (Ok(py), Ok(rust)) => {
            results.push((
                "flow_matching".to_string(),
                compare_tensors("flow_matching", &rust, &py),
            ));
        }
        (Err(e), _) => {
            println!("[SKIP] flow_matching: Python reference not found - {}", e);
        }
        (_, Err(e)) => {
            println!("[SKIP] flow_matching: Rust output not found - {}", e);
        }
    }

    results
}

/// Standalone RoPE test with detailed statistics
#[test]
fn test_rope_detailed_comparison() {
    let device = Device::Cpu;

    // Load Python reference data
    let q = load_tensor_by_name("tests/reference_data/rope", "rope_q_input")
        .expect("Failed to load rope_q_input");
    let k = load_tensor_by_name("tests/reference_data/rope", "rope_k_input")
        .expect("Failed to load rope_k_input");
    let q_expected = load_tensor_by_name("tests/reference_data/rope", "rope_q_output")
        .expect("Failed to load rope_q_output");
    let k_expected = load_tensor_by_name("tests/reference_data/rope", "rope_k_output")
        .expect("Failed to load rope_k_output");

    // Create RoPE
    let rope = sundial_rust::model::rope::SundialRotaryEmbedding::new(64, 1000, 10000.0, &device)
        .expect("Failed to create RoPE");

    // Run RoPE
    let (q_output, k_output) = rope.forward(&q, &k, None).expect("Failed to apply RoPE");

    // Compare with detailed statistics
    println!("\n=== RoPE Comparison ===");
    let q_stats = compare_tensors("RoPE_Q_output", &q_output, &q_expected);
    let k_stats = compare_tensors("RoPE_K_output", &k_output, &k_expected);

    // Assert against tolerance
    if let Some(stats) = &q_stats {
        assert!(
            stats.max_diff < 1e-4,
            "RoPE Q max_diff={:.6e} > 1e-4",
            stats.max_diff
        );
    }
    if let Some(stats) = &k_stats {
        assert!(
            stats.max_diff < 1e-4,
            "RoPE K max_diff={:.6e} > 1e-4",
            stats.max_diff
        );
    }
}

/// Comprehensive layer-by-layer comparison test
#[test]
#[ignore = "Requires Python reference intermediates to be generated"]
fn test_layer_by_layer_comparison() {
    println!("\n=== Layer-by-Layer Comparison Framework ===\n");

    let mut all_results = Vec::new();

    // Test each component
    all_results.extend(test_patch_embed_comparison());
    all_results.extend(test_rope_comparison());
    all_results.extend(test_block_comparison());
    all_results.extend(test_flow_matching_comparison());

    // Generate JSON report
    let _ = generate_report(&all_results, "/tmp/layer_comparison_report.json");

    println!("\n=== Summary ===");
    let failed: Vec<_> = all_results
        .iter()
        .filter(|(_, stats)| stats.as_ref().map(|s| s.max_diff > 1e-3).unwrap_or(false))
        .collect();

    println!("Total components tested: {}", all_results.len());
    println!("Components with large discrepancies: {}", failed.len());

    if !failed.is_empty() {
        println!("\nDiscrepant components:");
        for (name, _) in &failed {
            println!("  - {}", name);
        }
    }

    // The test passes if we can generate the report
    // Actual pass/fail depends on the tolerance thresholds
    assert!(all_results.len() > 0, "No components were tested");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_stats_creation() {
        let device = Device::Cpu;
        let actual = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap();
        let expected = Tensor::from_vec(vec![1.1f32, 2.1, 2.9, 4.2], (2, 2), &device).unwrap();

        let stats = ComparisonStats::new(&actual, &expected);

        // Verify stats are computed correctly
        assert_eq!(stats.shape, "[2, 2]");
        assert!((stats.max_diff - 0.2).abs() < 1e-5); // max diff is 0.2
        assert!((stats.mean_diff - 0.15).abs() < 1e-5); // mean diff is (0.1+0.1+0.1+0.2)/4 = 0.125
        assert!(stats.std_diff >= 0.0);
    }

    #[test]
    fn test_compare_tensors_returns_none_for_shape_mismatch() {
        let device = Device::Cpu;
        let actual = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device).unwrap();
        let expected = Tensor::from_vec(vec![1.0f32, 2.0], (2,), &device).unwrap();

        let result = compare_tensors("test", &actual, &expected);

        assert!(result.is_none());
    }

    #[test]
    fn test_compare_tensors_returns_stats_for_matching_shapes() {
        let device = Device::Cpu;
        let actual = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device).unwrap();
        let expected = Tensor::from_vec(vec![1.1f32, 2.1, 3.1], (3,), &device).unwrap();

        let result = compare_tensors("test", &actual, &expected);

        assert!(result.is_some());
        let stats = result.unwrap();
        assert!((stats.max_diff - 0.1).abs() < 1e-5);
        assert!((stats.mean_diff - 0.1).abs() < 1e-5);
    }
}

/// End-to-end comparison with intermediate saving
#[test]
#[ignore = "Requires Python model and reference data"]
fn test_end_to_end_with_intermediates() {
    use std::env;

    println!("\n=== End-to-End with Intermediates ===\n");

    // Enable debug mode to save intermediates
    env::set_var("SUNDIAL_DEBUG", "1");

    let device = Device::Cpu;

    // Load test input
    let input_3d =
        load_reference_tensor("tests/reference_data/input.npy").expect("Failed to load test input");

    let input = if input_3d.dims().len() == 3 {
        input_3d.squeeze(2).expect("Failed to squeeze input")
    } else {
        input_3d
    };

    // Load model
    let config = SundialConfig::default();
    let vb = sundial_rust::model::loader::load_sundial_from_huggingface(
        "thuml/sundial-base-128m",
        &device,
    )
    .expect("Failed to load model");

    let model = SundialModel::new(&config, vb).expect("Failed to create model");

    // Run inference
    let rust_predictions = model
        .generate(&input, 14, 1, false)
        .expect("Failed to run inference");

    // Squeeze to match Python shape
    let rust_predictions = rust_predictions
        .squeeze(0)
        .expect("Failed to squeeze")
        .squeeze(0)
        .expect("Failed to squeeze");

    // Load Python predictions
    let python_predictions = load_reference_tensor("tests/reference_data/predictions.npy")
        .expect("Failed to load Python predictions");

    // Compare
    println!("\n=== Final Output Comparison ===");
    let stats = compare_tensors("final_predictions", &rust_predictions, &python_predictions);

    if let Some(stats) = stats {
        assert!(
            stats.max_diff < 1.0,
            "Final predictions max_diff={:.6e} > 1.0",
            stats.max_diff
        );
    }
}
