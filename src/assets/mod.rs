//! Asset module for embedded model weights.
//!
//! This module provides compile-time embedded assets for the Sundial model,
//! including compressed weights, configuration, and integrity hashes.
//!
//! ## Overview
//!
//! The assets module embeds the following at compile time:
//! - Compressed model weights (`WEIGHTS_COMPRESSED`)
//! - Model configuration JSON (`CONFIG_JSON`)
//! - SHA256 hash for integrity verification (`MODEL_SHA256`)
//!
//! ## Usage
//!
//! ```no_run
//! use sundial_rust::assets::{load_config, MODEL_SHA256};
//!
//! // Load configuration
//! let config = load_config().expect("Failed to load config");
//!
//! // Verify hash (would be done in weight loader)
//! println!("Model hash: {}", MODEL_SHA256);
//! ```

use std::error::Error;
use std::fmt;

/// Custom error types for asset operations
#[derive(Debug)]
pub enum AssetError {
    /// Failed to parse configuration JSON
    ConfigParse(String),
    /// Failed to decompress weights
    Decompression(String),
    /// Hash verification failed
    HashMismatch { expected: String, actual: String },
}

impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssetError::ConfigParse(msg) => write!(f, "Config parse error: {}", msg),
            AssetError::Decompression(msg) => write!(f, "Decompression error: {}", msg),
            AssetError::HashMismatch { expected, actual } => {
                write!(f, "Hash mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl Error for AssetError {}

/// Embedded compressed model weights
///
/// This constant contains the gzip-compressed model weights embedded at compile time
/// via the build.rs script. Use [extract_weights()](crate::weights::loader::extract)
/// to decompress to a temporary location for model loading.
pub const WEIGHTS_COMPRESSED: &[u8] = include_bytes!(env!("WEIGHTS_PATH"));

/// Embedded model configuration JSON
///
/// This constant contains the raw config.json file embedded at compile time.
/// Parse this to obtain [SundialConfig](crate::model::SundialConfig).
pub const CONFIG_JSON: &[u8] = include_bytes!("../../weights/config.json");

/// SHA256 hash of the original (uncompressed) model weights
///
/// This hash is computed at build time and used to verify weight integrity
/// after extraction. The hash corresponds to the raw model.safetensors file.
pub const MODEL_SHA256: &str = env!("MODEL_SHA256");

/// Parse the embedded configuration JSON into a SundialConfig
///
/// # Returns
/// * `Ok(SundialConfig)` if parsing succeeds
/// * `Err(AssetError)` if JSON parsing fails
pub fn load_config() -> Result<crate::model::SundialConfig, AssetError> {
    let config_str = String::from_utf8(CONFIG_JSON.to_vec())
        .map_err(|e| AssetError::ConfigParse(e.to_string()))?;

    serde_json::from_str(&config_str).map_err(|e| AssetError::ConfigParse(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weights_compressed_is_non_empty() {
        assert!(
            !WEIGHTS_COMPRESSED.is_empty(),
            "Compressed weights should not be empty"
        );
    }

    #[test]
    fn test_config_json_is_non_empty() {
        assert!(!CONFIG_JSON.is_empty(), "Config JSON should not be empty");
    }

    #[test]
    fn test_model_sha256_format() {
        // SHA256 hashes are 64 hex characters
        assert_eq!(
            MODEL_SHA256.len(),
            64,
            "SHA256 hash should be 64 hex characters"
        );
        assert!(
            MODEL_SHA256.chars().all(|c| c.is_ascii_hexdigit()),
            "SHA256 hash should contain only hex digits"
        );
    }

    #[test]
    fn test_load_config_succeeds() {
        let config = load_config().expect("Should be able to load config");
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.input_token_len, 16);
    }
}
