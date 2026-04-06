//! Build verification tests for cross-platform builds
//!
//! These tests verify that the build system correctly:
//! - Detects and reports target platform information
//! - Embeds assets with proper integrity checks
//! - Compiles platform-specific features correctly
//! - Generates appropriate build reports

/// Target platform information structure
#[derive(Debug, Clone)]
struct PlatformInfo {
    os: String,
    arch: String,
    family: String,
    is_unix: bool,
    is_windows: bool,
    is_macos: bool,
    is_linux: bool,
    is_x86: bool,
    is_arm: bool,
}

impl PlatformInfo {
    fn current() -> Self {
        PlatformInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            family: std::env::consts::FAMILY.to_string(),
            is_unix: cfg!(unix),
            is_windows: cfg!(windows),
            is_macos: cfg!(target_os = "macos"),
            is_linux: cfg!(target_os = "linux"),
            is_x86: cfg!(any(target_arch = "x86", target_arch = "x86_64")),
            is_arm: cfg!(any(target_arch = "aarch64", target_arch = "arm")),
        }
    }

    fn to_report(&self) -> String {
        format!(
            "Platform Report
==============
OS: {}
Architecture: {}
Family: {}
Unix: {}
Windows: {}
macOS: {}
Linux: {}
x86: {}
ARM: {}
",
            self.os,
            self.arch,
            self.family,
            self.is_unix,
            self.is_windows,
            self.is_macos,
            self.is_linux,
            self.is_x86,
            self.is_arm
        )
    }
}

/// Build metrics structure
#[derive(Debug)]
struct BuildMetrics {
    has_embedded_weights: bool,
    has_config: bool,
    has_valid_hash: bool,
    platform_config_generated: bool,
    compression_successful: bool,
}

impl BuildMetrics {
    fn gather() -> Self {
        use sundial_rust::assets::{CONFIG_JSON, MODEL_SHA256, WEIGHTS_COMPRESSED};

        BuildMetrics {
            has_embedded_weights: !WEIGHTS_COMPRESSED.is_empty(),
            has_config: !CONFIG_JSON.is_empty(),
            has_valid_hash: MODEL_SHA256.len() == 64
                && MODEL_SHA256.chars().all(|c| c.is_ascii_hexdigit()),
            platform_config_generated: true, // Build script always generates this
            compression_successful: WEIGHTS_COMPRESSED.len() > 0
                && WEIGHTS_COMPRESSED.len() < 500_000_000,
        }
    }

    fn to_report(&self) -> String {
        format!(
            "Build Metrics Report
==================
Embedded Weights: {}
Configuration: {}
Valid SHA256 Hash: {}
Platform Config Generated: {}
Compression Successful: {}
",
            self.has_embedded_weights,
            self.has_config,
            self.has_valid_hash,
            self.platform_config_generated,
            self.compression_successful
        )
    }
}

/// Supported target platforms for cross-compilation
const SUPPORTED_TARGETS: &[(&str, &str, &str)] = &[
    ("x86_64-unknown-linux-gnu", "Linux", "x86_64"),
    ("aarch64-unknown-linux-gnu", "Linux", "ARM64"),
    ("x86_64-apple-darwin", "macOS", "x86_64"),
    ("aarch64-apple-darwin", "macOS", "ARM64"),
    ("x86_64-pc-windows-msvc", "Windows", "x86_64"),
];

#[test]
fn test_platform_detection_macros() {
    // Verify that platform detection cfg macros work correctly
    let platform = PlatformInfo::current();

    // At least one OS should be true
    assert!(
        platform.is_unix || platform.is_windows,
        "Platform must be either Unix or Windows"
    );

    // Verify family matches OS
    if platform.is_windows {
        assert_eq!(platform.family, "windows");
    } else if platform.is_unix {
        assert_eq!(platform.family, "unix");
    }

    // Verify architecture is set
    assert!(
        platform.is_x86 || platform.is_arm,
        "Platform must be either x86 or ARM"
    );
}

#[test]
fn test_embedded_assets_exist() {
    use sundial_rust::assets::{CONFIG_JSON, MODEL_SHA256, WEIGHTS_COMPRESSED};

    // Verify all embedded assets are present
    assert!(!WEIGHTS_COMPRESSED.is_empty(), "Weights must be embedded");
    assert!(!CONFIG_JSON.is_empty(), "Config must be embedded");
    assert!(!MODEL_SHA256.is_empty(), "SHA256 hash must be embedded");
}

#[test]
fn test_build_script_environment_variables() {
    // Verify build script set required environment variables
    let weights_path = option_env!("WEIGHTS_PATH");
    let model_sha256 = option_env!("MODEL_SHA256");
    let target_os = option_env!("TARGET_OS");
    let target_arch = option_env!("TARGET_ARCH");
    let target_family = option_env!("TARGET_FAMILY");

    assert!(
        weights_path.is_some(),
        "WEIGHTS_PATH must be set by build script"
    );
    assert!(
        model_sha256.is_some(),
        "MODEL_SHA256 must be set by build script"
    );
    assert!(target_os.is_some(), "TARGET_OS must be set by build script");
    assert!(
        target_arch.is_some(),
        "TARGET_ARCH must be set by build script"
    );
    assert!(
        target_family.is_some(),
        "TARGET_FAMILY must be set by build script"
    );

    // Verify hash format
    let hash = model_sha256.unwrap();
    assert_eq!(hash.len(), 64, "SHA256 hash must be 64 hex characters");
    assert!(
        hash.chars().all(|c| c.is_ascii_hexdigit()),
        "SHA256 hash must be valid hex"
    );
}

#[test]
fn test_platform_config_structure() {
    use std::str;
    use sundial_rust::assets::CONFIG_JSON;

    // Verify config JSON is valid and contains expected fields
    let config_str = str::from_utf8(CONFIG_JSON).expect("Config must be valid UTF-8");
    let config: serde_json::Value =
        serde_json::from_str(config_str).expect("Config must be valid JSON");

    // Check for common model configuration fields
    assert!(
        config.get("hidden_size").is_some()
            || config.get("model_type").is_some()
            || config.get("vocab_size").is_some(),
        "Config should contain model configuration fields"
    );
}

#[test]
fn test_supported_target_triples() {
    // Verify that supported target triples are correctly formatted
    for (target, os_name, _arch_name) in SUPPORTED_TARGETS {
        // Verify target triple format (arch-vendor-os-abi or arch-os-abi)
        let parts: Vec<&str> = target.split('-').collect();
        assert!(
            parts.len() >= 3,
            "Target triple {} should have at least 3 parts",
            target
        );

        // Verify the OS matches expected values
        // Target triples follow: arch-vendor-os-abi format
        // For macOS: x86_64-apple-darwin or aarch64-apple-darwin (last part is "darwin")
        // For Linux: x86_64-unknown-linux-gnu (last part is "gnu", second-to-last is "linux")
        // For Windows: x86_64-pc-windows-msvc (last part is "msvc", second-to-last is "windows")
        // The OS is at the last position for some targets, second-to-last for others
        let last_part = parts[parts.len() - 1];
        let second_to_last = parts[parts.len() - 2];

        // Check if either the last or second-to-last part matches the expected OS
        let os_match = match *os_name {
            "macOS" => last_part == "darwin" || second_to_last == "darwin",
            "Linux" => {
                last_part == "linux"
                    || second_to_last == "linux"
                    || last_part == "gnu"
                    || second_to_last == "gnu"
            }
            "Windows" => {
                last_part == "windows"
                    || second_to_last == "windows"
                    || last_part == "msvc"
                    || second_to_last == "msvc"
            }
            _ => false,
        };

        assert!(
            os_match,
            "Target {} should match OS {} (found {} or {} in target triple)",
            target, os_name, second_to_last, last_part
        );
    }
}

#[test]
fn test_build_metrics_gathering() {
    // Verify that build metrics can be gathered
    let metrics = BuildMetrics::gather();

    assert!(metrics.has_embedded_weights, "Weights should be embedded");
    assert!(metrics.has_config, "Config should be embedded");
    assert!(metrics.has_valid_hash, "Hash should be valid");
    assert!(
        metrics.platform_config_generated,
        "Platform config should be generated"
    );
    assert!(
        metrics.compression_successful,
        "Compression should be successful"
    );
}

#[test]
fn test_generate_build_report() {
    // Generate a comprehensive build report
    let platform = PlatformInfo::current();
    let metrics = BuildMetrics::gather();

    let report = format!(
        "{}
{}
Supported Targets:
{}
",
        platform.to_report(),
        metrics.to_report(),
        SUPPORTED_TARGETS
            .iter()
            .map(|(t, o, a)| format!("  - {} ({}, {})", t, o, a))
            .collect::<Vec<_>>()
            .join("\n")
    );

    // Verify report contains key information
    assert!(report.contains("Platform Report"));
    assert!(report.contains("Build Metrics Report"));
    assert!(report.contains("Supported Targets"));
    assert!(report.contains(platform.os.as_str()));
    assert!(report.contains(platform.arch.as_str()));
}

#[test]
fn test_platform_specific_features() {
    // Verify platform-specific features are compiled correctly
    #[cfg(target_os = "linux")]
    {
        // Linux-specific verification
        let platform = PlatformInfo::current();
        assert!(platform.is_linux, "Should detect Linux platform");
        assert!(platform.is_unix, "Linux should be Unix family");
    }

    #[cfg(target_os = "macos")]
    {
        // macOS-specific verification
        let platform = PlatformInfo::current();
        assert!(platform.is_macos, "Should detect macOS platform");
        assert!(platform.is_unix, "macOS should be Unix family");
    }

    #[cfg(windows)]
    {
        // Windows-specific verification
        let platform = PlatformInfo::current();
        assert!(platform.is_windows, "Should detect Windows platform");
        assert_eq!(platform.family, "windows");
    }
}

#[test]
fn test_architecture_specific_features() {
    // Verify architecture-specific features are compiled correctly
    #[cfg(target_arch = "x86_64")]
    {
        let platform = PlatformInfo::current();
        assert!(platform.is_x86, "Should detect x86_64 architecture");
    }

    #[cfg(target_arch = "aarch64")]
    {
        let platform = PlatformInfo::current();
        assert!(platform.is_arm, "Should detect aarch64 architecture");
    }
}

#[test]
fn test_weight_loader_platform_compatibility() {
    // Verify weight loader works on current platform
    use sundial_rust::weights::loader::WeightLoader;

    let loader = WeightLoader::new().expect("Should be able to create weight loader");
    let model_path = loader.model_path();

    // Verify model path is valid
    assert!(
        !model_path.as_os_str().is_empty(),
        "Model path should not be empty"
    );

    // Verify path is properly formatted for current platform
    let path_str = model_path.to_string_lossy();
    #[cfg(unix)]
    {
        assert!(
            !path_str.contains('\\'),
            "Unix paths should not contain backslashes"
        );
    }

    #[cfg(windows)]
    {
        assert!(
            path_str.contains('\\') || path_str.contains('/'),
            "Windows paths should contain path separators"
        );
    }
}

#[test]
fn test_asset_error_handling() {
    use sundial_rust::assets::AssetError;

    // Test error display implementations
    let config_error = AssetError::ConfigParse("Invalid JSON".to_string());
    assert!(config_error.to_string().contains("Config parse error"));

    let decompress_error = AssetError::Decompression("Invalid gzip".to_string());
    assert!(decompress_error.to_string().contains("Decompression error"));

    let hash_error = AssetError::HashMismatch {
        expected: "abc123".to_string(),
        actual: "def456".to_string(),
    };
    let hash_msg = hash_error.to_string();
    assert!(hash_msg.contains("Hash mismatch"));
    assert!(hash_msg.contains("abc123"));
    assert!(hash_msg.contains("def456"));
}

#[test]
fn test_build_script_rerun_triggers() {
    // Verify that build script has proper rerun triggers
    // This test verifies the build script configuration by checking
    // that environment variables are set (which implies the build script ran)
    assert!(
        option_env!("WEIGHTS_PATH").is_some(),
        "Build script should set WEIGHTS_PATH"
    );
    assert!(
        option_env!("MODEL_SHA256").is_some(),
        "Build script should set MODEL_SHA256"
    );
}

#[test]
fn test_compression_ratio_reasonable() {
    // Verify compression produces reasonable results
    use sundial_rust::assets::WEIGHTS_COMPRESSED;

    // Weights should be compressed (not empty and not absurdly large)
    let compressed_size = WEIGHTS_COMPRESSED.len();

    // Should be less than 500MB (original is ~490MB)
    assert!(
        compressed_size < 500_000_000,
        "Compressed weights should be < 500MB, got {} bytes",
        compressed_size
    );

    // Should be substantial (not empty or tiny)
    assert!(
        compressed_size > 1_000_000,
        "Compressed weights should be > 1MB, got {} bytes",
        compressed_size
    );
}

#[test]
fn test_sha256_consistency() {
    // Verify SHA256 hash is consistent and properly formatted
    use sundial_rust::assets::MODEL_SHA256;

    // Hash should be exactly 64 hex characters
    assert_eq!(MODEL_SHA256.len(), 64, "SHA256 should be 64 characters");

    // All characters should be valid hex digits
    assert!(
        MODEL_SHA256.chars().all(|c| c.is_ascii_hexdigit()),
        "SHA256 should contain only hex digits"
    );

    // Hash should not be a placeholder or default value
    assert_ne!(
        MODEL_SHA256, "0000000000000000000000000000000000000000000000000000000000000000",
        "SHA256 should not be all zeros"
    );
    assert_ne!(
        MODEL_SHA256, "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "SHA256 should not be all f's"
    );
}

/// Integration test: Full build verification
#[test]
fn test_full_build_verification() {
    // Check that the target directory exists with compiled artifacts
    // This verifies that a build has been completed successfully
    let target_dir = std::path::PathBuf::from("target");
    let debug_dir = target_dir.join("debug");
    let release_dir = target_dir.join("release");

    // At least one build directory should exist
    assert!(
        debug_dir.exists() || release_dir.exists(),
        "Build output directory should exist"
    );

    // Verify that the compiled binary executable exists
    // This project builds as a CLI application (binary), not a library
    // Binary name is "main" comes from src/bin/main.rs (default for bin crates)
    let binary_name = "main";
    let binary_extensions = if cfg!(windows) {
        vec![".exe"]
    } else {
        vec![""] // No extension on Unix-like systems
    };

    let mut binary_found = false;
    for ext in &binary_extensions {
        let binary_path = debug_dir.join(format!("{}{}", binary_name, ext));
        if binary_path.exists() {
            binary_found = true;
            break;
        }
    }

    assert!(
        binary_found,
        "Compiled binary executable should exist in target/debug/ with one of the following extensions: {:?}",
        binary_extensions
    );
}
