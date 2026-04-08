//! Testing infrastructure for Sundial Rust
//!
//! This module provides utilities for correctness testing against Python references.
//! It includes:
//! - Reference tensor loading from .npy files
//! - Tensor comparison assertions with configurable tolerances
//! - Debug utilities for analyzing discrepancies

pub mod assertions;
pub mod reference_loader;

pub use assertions::{
    assert_tensor_close, assert_tensor_exact, assert_tensor_mape, assert_tensor_relaxed,
    compute_mape, compute_max_diff, compute_mean_diff, print_comparison_stats,
};
pub use reference_loader::{
    load_reference_tensor, load_reference_tensors_from_dir, load_tensor_by_name, save_tensor_to_npy,
};
