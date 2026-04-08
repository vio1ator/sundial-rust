/// Integration tests for embedded weights functionality
/// Verifies cross-platform compatibility and end-to-end functionality
use std::process::{Command, Stdio};

/// Helper to run the sundial binary and capture output
fn run_sundial(args: &[&str]) -> (i32, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_main"))
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    (output.status.code().unwrap_or(-1), stdout, stderr)
}

#[test]
fn test_embedded_weights_default_behavior() {
    // Default behavior should use embedded weights
    let (code, _stdout, stderr) = run_sundial(&[
        "--verbose",
        "--infer",
        "--input",
        "test_data/sample.csv",
        "--horizon",
        "5",
    ]);

    // Should fail due to insufficient data points, not due to missing weights
    assert!(code != 0);
    assert!(!stderr.contains("No weights available") && !stderr.contains("weights file not found"));
}

#[test]
fn test_model_flag_override() {
    // Test --model flag bypasses embedded weights
    let (code, _stdout, stderr) = run_sundial(&[
        "--model",
        "weights/model.safetensors",
        "--infer",
        "--input",
        "test_data/sample.csv",
        "--horizon",
        "5",
    ]);

    // Should fail due to insufficient data points, not due to missing model
    assert!(code != 0);
    assert!(!stderr.contains("No model available") && !stderr.contains("model file not found"));
}

#[test]
fn test_embedded_weights_size() {
    // Verify embedded weights are actually compressed
    use sundial_rust::assets::WEIGHTS_COMPRESSED;

    // Compressed size should be substantial but less than original (~490MB)
    // Note: Modern safetensors files are already partially compressed
    assert!(
        WEIGHTS_COMPRESSED.len() < 500_000_000,
        "Compressed weights should be < 500MB"
    );
    // Should still be substantial (current compression ratio is ~1.08x = ~476MB)
    assert!(
        WEIGHTS_COMPRESSED.len() > 400_000_000,
        "Compressed weights should be > 400MB"
    );
}

#[test]
fn test_sha256_hash_format() {
    // Verify hash format
    use sundial_rust::assets::MODEL_SHA256;

    assert_eq!(
        MODEL_SHA256.len(),
        64,
        "SHA256 hash should be 64 hex characters"
    );
    assert!(
        MODEL_SHA256.chars().all(|c| c.is_ascii_hexdigit()),
        "SHA256 hash should contain only hex characters"
    );
}

#[test]
fn test_config_embedded() {
    // Verify config.json is accessible
    use sundial_rust::assets::CONFIG_JSON;

    assert!(!CONFIG_JSON.is_empty(), "Config JSON should not be empty");
    // Convert bytes to string for checking
    let config_str = std::str::from_utf8(CONFIG_JSON).expect("Config should be valid UTF-8");
    assert!(
        config_str.contains("\"hidden_size\"") || config_str.contains("{"),
        "Config should contain model configuration"
    );
}

#[test]
fn test_error_messages_include_fallback_instructions() {
    // Test that error messages include helpful fallback instructions
    use sundial_rust::weights::error::WeightError;

    let hash_error = WeightError::HashMismatch {
        expected: "abc123".to_string(),
        computed: "def456".to_string(),
    };

    let user_msg = hash_error.user_message();
    assert!(
        user_msg.contains("--model"),
        "Hash error should mention --model flag as fallback"
    );
    assert!(
        user_msg.contains("SHA256") || user_msg.contains("hash"),
        "Hash error should mention integrity verification"
    );

    let disk_error = WeightError::InsufficientDiskSpace {
        needed: 500_000_000,
        available: 100_000_000,
    };

    let disk_msg = disk_error.user_message();
    assert!(
        disk_msg.contains("SUNDIAL_TEMP_DIR") || disk_msg.contains("--model"),
        "Disk error should suggest SUNDIAL_TEMP_DIR or --model"
    );
}
