//! Weight loading and extraction module.
//!
//! This module provides functionality to load and extract compressed model weights
//! from the embedded assets at runtime. The weights are decompressed to a temporary
//! directory with secure permissions before being passed to the model loader.
//!
//! ## Overview
//!
//! The weight loader provides:
//! - Extraction of compressed weights to temporary storage
//! - SHA256 hash verification for integrity checking
//! - Support for external weight files via environment variables
//!
//! ## Usage
//!
//! ```no_run
//! use sundial_rust::weights::loader::{get_model_path, get_config_path};
//! use candle_core::Device;
//!
//! // Get paths to extracted weights
//! let model_path = get_model_path().expect("Failed to get model path");
//! let config_path = get_config_path().expect("Failed to get config path");
//!
//! // Load the model using these paths
//! ```

pub mod error;
pub mod loader;

// Re-export common types for convenience
pub use error::WeightError;
pub use loader::{get_config_path, get_model_path, WeightLoader};
