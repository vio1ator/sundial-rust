//! Custom error types for the weights module.
//!
//! This module provides specialized error types for weight loading, extraction,
//! and verification operations. It uses `thiserror` for derive-based error
//! implementation and integrates with `anyhow` for application-level errors.

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during weight loading and extraction
#[derive(Error, Debug)]
pub enum WeightError {
    /// Failed to open a weights file
    #[error("Failed to open weights file: {0}")]
    FileOpenError(String),

    /// Failed to read a weights file
    #[error("Failed to read weights file: {0}")]
    FileReadError(String),

    /// Failed to write a weights file
    #[error("Failed to write weights file: {0}")]
    FileWriteError(String),

    /// Failed to create a directory
    #[error("Failed to create directory: {0}")]
    DirCreationError(String),

    /// Failed to decompress weights
    #[error("Failed to decompress weights: {0}")]
    DecompressionError(String),

    /// Failed to compress weights (build-time)
    #[error("Failed to compress weights: {0}")]
    CompressionError(String),

    /// SHA256 hash mismatch detected
    #[error(
        "SHA256 hash mismatch!\n  Expected:   {expected}\n  Computed:   {computed}\n\nThis indicates corrupted or tampered weights.\nFallback: Use --model flag with a valid weights file"
    )]
    HashMismatch { expected: String, computed: String },

    /// Insufficient disk space for extraction
    #[error(
        "Insufficient disk space: need {needed} bytes, available {available} bytes\n\nFallback: Use --model flag with a valid weights file or free up disk space"
    )]
    InsufficientDiskSpace { needed: u64, available: u64 },

    /// Permission denied when accessing weights
    #[error(
        "Permission denied: {0}\n\nThis may be due to restrictive file permissions on the temp directory.\nFallback: Set SUNDIAL_TEMP_DIR to a writable location or use --model flag"
    )]
    PermissionDenied(String),

    /// Custom temp directory not found or invalid
    #[error("Custom temp directory not found or invalid: {0}\n\nFallback: Set SUNDIAL_TEMP_DIR to a valid writable directory or use --model flag")]
    TempDirNotFound(PathBuf),

    /// Failed to set file/directory permissions
    #[error("Failed to set permissions: {0}")]
    SetPermissionsError(#[from] std::io::Error),

    /// Config file not found
    #[error("Config file not found: {0}\n\nFallback: Set SUNDIAL_CONFIG_PATH to a valid config file or use --model flag")]
    ConfigNotFound(PathBuf),

    /// No weights available (neither embedded nor external)
    #[error(
        "No weights available.\n\nTo use embedded weights (default): sundial --infer --input <data>\nTo use custom weights: sundial --infer --model <path> --input <data>\nOr set SUNDIAL_MODEL_PATH environment variable"
    )]
    NoWeightsAvailable,

    /// Weights extraction failed
    #[error(
        "Failed to extract weights: {0}\n\nDebugging steps:\n1. Check available disk space (need at least 500MB)\n2. Verify temp directory is writable\n3. Check SUNDIAL_TEMP_DIR environment variable\n\nFallback: Use --model flag with a valid .safetensors file"
    )]
    ExtractionFailed(String),

    /// Weights verification failed
    #[error(
        "Failed to verify weights integrity: {0}\n\nDebugging steps:\n1. Verify weights file is not corrupted\n2. Check SHA256 hash matches expected value\n3. Ensure file permissions allow reading\n\nFallback: Use --model flag with a valid .safetensors file"
    )]
    VerificationFailed(String),
}

impl WeightError {
    /// Get a user-friendly error message for display
    pub fn user_message(&self) -> String {
        match self {
            WeightError::HashMismatch { .. } => {
                "⚠️  Weights file integrity check failed!\n\nThe downloaded or embedded weights have an invalid SHA256 hash,\nwhich could indicate corruption or tampering.\n\nSolutions:\n1. Re-download the model weights from a trusted source\n2. Use the --model flag to specify a different weights file\n3. Check your network connection if downloading weights\n\nExample: sundial --infer --model /path/to/valid/weights.safetensors --input data.csv".to_string()
            }
            WeightError::InsufficientDiskSpace { needed, available } => {
                format!(
                    "⚠️  Insufficient disk space!\n\nNeed:      {} bytes ({:.2} MB)\nAvailable:  {} bytes ({:.2} MB)\n\nSolutions:\n1. Free up disk space by deleting unnecessary files\n2. Set SUNDIAL_TEMP_DIR to a directory with more space\n3. Use the --model flag to specify weights on a different drive\n\nExample: SUNDIAL_TEMP_DIR=/mnt/large_drive/sundial sundial --infer --input data.csv",
                    needed,
                    *needed as f64 / 1_000_000.0,
                    available,
                    *available as f64 / 1_000_000.0
                )
            }
            WeightError::PermissionDenied(msg) => {
                format!(
                    "⚠️  Permission denied: {}\n\nThe weights extraction failed due to insufficient permissions.\n\nSolutions:\n1. Set SUNDIAL_TEMP_DIR to a writable directory\n2. Check file permissions on the temp directory\n3. Use the --model flag to specify an already-extracted weights file\n\nExample: SUNDIAL_TEMP_DIR=/tmp/sundial sundial --infer --input data.csv",
                    msg
                )
            }
            WeightError::TempDirNotFound(path) => {
                format!(
                    "⚠️  Invalid temp directory: {:?}\n\nThe specified SUNDIAL_TEMP_DIR does not exist or is not accessible.\n\nSolution:\n1. Create the directory: mkdir -p {:?}\n2. Set permissions: chmod 700 {:?}\n3. Or use default temp directory by unsetting SUNDIAL_TEMP_DIR\n4. Use --model flag to bypass extraction\n\nExample: SUNDIAL_TEMP_DIR=/tmp/sundial sundial --infer --input data.csv",
                    path, path, path
                )
            }
            WeightError::DecompressionError(_) => {
                "⚠️  Failed to decompress weights file!\n\nThe embedded or specified weights file appears to be corrupted.\n\nSolutions:\n1. Re-download the model weights\n2. Check if the weights file is complete\n3. Use the --model flag to specify a different weights file\n\nExample: sundial --infer --model /path/to/valid/weights.safetensors --input data.csv".to_string()
            }
            WeightError::ExtractionFailed(msg) => {
                format!(
                    "⚠️  Failed to extract weights: {}\n\nDebugging steps:\n1. Check available disk space (need at least 500MB)\n2. Verify temp directory is writable\n3. Check SUNDIAL_TEMP_DIR environment variable\n4. Ensure no other process is locking the files\n\nFallback: Use --model flag to specify an already-extracted weights file\n\nExample: sundial --infer --model /path/to/weights.safetensors --input data.csv",
                    msg
                )
            }
            WeightError::VerificationFailed(msg) => {
                format!(
                    "⚠️  Weights verification failed: {}\n\nThe extracted weights failed integrity verification.\n\nDebugging steps:\n1. Verify the weights file is not corrupted\n2. Check if the SHA256 hash matches the expected value\n3. Ensure the file was downloaded completely\n\nFallback: Use --model flag with a valid .safetensors file\n\nExample: sundial --infer --model /path/to/valid/weights.safetensors --input data.csv",
                    msg
                )
            }
            _ => self.to_string(),
        }
    }

    /// Create a permission denied error with SUNDIAL_TEMP_DIR suggestion
    pub fn permission_denied_suggestion(path: &PathBuf) -> Self {
        WeightError::PermissionDenied(format!(
            "Cannot access path: {:?}\n\nTry setting SUNDIAL_TEMP_DIR to a writable location:\n  export SUNDIAL_TEMP_DIR=/tmp/sundial\n\nOr use the --model flag to specify an existing weights file.",
            path
        ))
    }

    /// Create an extraction failed error
    pub fn extraction_failed(msg: impl Into<String>) -> Self {
        WeightError::ExtractionFailed(msg.into())
    }

    /// Create a verification failed error
    pub fn verification_failed(msg: impl Into<String>) -> Self {
        WeightError::VerificationFailed(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_mismatch_user_message() {
        let err = WeightError::HashMismatch {
            expected: "abc123".to_string(),
            computed: "def456".to_string(),
        };
        let msg = err.user_message();
        assert!(msg.contains("SHA256"));
        assert!(msg.contains("--model"));
    }

    #[test]
    fn test_insufficient_disk_space_user_message() {
        let err = WeightError::InsufficientDiskSpace {
            needed: 500_000_000,
            available: 100_000_000,
        };
        let msg = err.user_message();
        assert!(msg.contains("Insufficient disk space"));
        assert!(msg.contains("SUNDIAL_TEMP_DIR"));
    }

    #[test]
    fn test_permission_denied_user_message() {
        let err = WeightError::PermissionDenied("access".to_string());
        let msg = err.user_message();
        assert!(msg.contains("Permission denied"));
        assert!(msg.contains("SUNDIAL_TEMP_DIR"));
    }
}
