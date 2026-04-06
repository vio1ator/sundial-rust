//! Tensor comparison assertions for correctness testing
//!
//! This module provides assertion functions to compare tensors against
//! Python reference implementations with configurable tolerances.

use anyhow::{bail, Result};
use candle_core::Tensor;

/// Compute the maximum absolute difference between two tensors
///
/// # Arguments
/// * `actual` - The tensor produced by the Rust implementation
/// * `expected` - The reference tensor from Python
///
/// # Returns
/// The maximum absolute difference as an f32 scalar
pub fn compute_max_diff(actual: &Tensor, expected: &Tensor) -> Result<f32> {
    let diff = actual.sub(expected)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    Ok(max_diff)
}

/// Compute the mean absolute difference between two tensors
///
/// # Arguments
/// * `actual` - The tensor produced by the Rust implementation
/// * `expected` - The reference tensor from Python
///
/// # Returns
/// The mean absolute difference as an f32 scalar
pub fn compute_mean_diff(actual: &Tensor, expected: &Tensor) -> Result<f32> {
    let diff = actual.sub(expected)?.abs()?;
    let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;
    Ok(mean_diff)
}

/// Assert that two tensors are close within a tolerance
///
/// This is the primary assertion function for correctness testing.
/// It checks that the maximum absolute difference between tensors
/// is within the specified tolerance.
///
/// # Arguments
/// * `actual` - The tensor produced by the Rust implementation
/// * `expected` - The reference tensor from Python
/// * `tolerance` - Maximum allowed absolute difference
/// * `name` - Descriptive name for error reporting
///
/// # Example
/// ```ignore
/// assert_tensor_close(&rust_output, &python_reference, 1e-4, "Attention output")?;
/// ```
pub fn assert_tensor_close(
    actual: &Tensor,
    expected: &Tensor,
    tolerance: f32,
    name: &str,
) -> Result<()> {
    // Check shapes match
    if actual.shape() != expected.shape() {
        bail!(
            "{}: shape mismatch - actual={:?}, expected={:?}",
            name,
            actual.shape(),
            expected.shape()
        );
    }

    // Compute differences
    let max_diff = compute_max_diff(actual, expected)?;
    let mean_diff = compute_mean_diff(actual, expected)?;

    // Check tolerance
    if max_diff > tolerance {
        bail!(
            "{}: max_diff={:.6e} > tolerance={:.6e}, mean_diff={:.6e}",
            name,
            max_diff,
            tolerance,
            mean_diff
        );
    }

    Ok(())
}

/// Assert that two tensors are very close (tight tolerance)
///
/// This is a convenience function for components that should match
/// almost exactly (e.g., RoPE, LayerNorm, MLP).
///
/// # Arguments
/// * `actual` - The tensor produced by the Rust implementation
/// * `expected` - The reference tensor from Python
/// * `name` - Descriptive name for error reporting
///
/// # Example
/// ```ignore
/// assert_tensor_exact(&rope_output, &python_reference, "RoPE output")?;
/// ```
pub fn assert_tensor_exact(actual: &Tensor, expected: &Tensor, name: &str) -> Result<()> {
    assert_tensor_close(actual, expected, 1e-5, name)
}

/// Assert that two tensors match with relaxed tolerance
///
/// This is useful for components with error accumulation or
/// numerical instability (e.g., full model output).
///
/// # Arguments
/// * `actual` - The tensor produced by the Rust implementation
/// * `expected` - The reference tensor from Python
/// * `name` - Descriptive name for error reporting
///
/// # Example
/// ```ignore
/// assert_tensor_relaxed(&model_output, &python_reference, 1.0, "Full model output")?;
/// ```
pub fn assert_tensor_relaxed(
    actual: &Tensor,
    expected: &Tensor,
    name: &str,
) -> Result<()> {
    assert_tensor_close(actual, expected, 1e-3, name)
}

/// Compute relative error (MAPE - Mean Absolute Percentage Error)
///
/// # Arguments
/// * `actual` - The tensor produced by the Rust implementation
/// * `expected` - The reference tensor from Python
///
/// # Returns
/// The mean absolute percentage error as a percentage (0-100)
pub fn compute_mape(actual: &Tensor, expected: &Tensor) -> Result<f32> {
    use candle_core::Device;
    
    let abs_diff = actual.sub(expected)?.abs()?;
    let abs_expected = expected.abs()?;
    
    // Avoid division by zero by creating a scalar tensor
    let device = Device::Cpu;
    let eps_tensor = Tensor::new(1e-8f32, &device)?;
    let abs_expected_safe = abs_expected.add(&eps_tensor)?;
    
    let relative_error = abs_diff.div(&abs_expected_safe)?;
    let mape = relative_error.mean_all()?.to_scalar::<f32>()? * 100.0;
    
    Ok(mape)
}

/// Assert that two tensors match within a percentage tolerance
///
/// # Arguments
/// * `actual` - The tensor produced by the Rust implementation
/// * `expected` - The reference tensor from Python
/// * `max_mape` - Maximum allowed MAPE (as percentage, e.g., 5.0 for 5%)
/// * `name` - Descriptive name for error reporting
pub fn assert_tensor_mape(
    actual: &Tensor,
    expected: &Tensor,
    max_mape: f32,
    name: &str,
) -> Result<()> {
    let mape = compute_mape(actual, expected)?;
    
    if mape > max_mape {
        bail!(
            "{}: MAPE={:.2}% > max_mape={:.2}%",
            name,
            mape,
            max_mape
        );
    }
    
    Ok(())
}

/// Print comparison statistics (for debugging)
///
/// # Arguments
/// * `actual` - The tensor produced by the Rust implementation
/// * `expected` - The reference tensor from Python
/// * `name` - Descriptive name
pub fn print_comparison_stats(actual: &Tensor, expected: &Tensor, name: &str) -> Result<()> {
    let max_diff = compute_max_diff(actual, expected)?;
    let mean_diff = compute_mean_diff(actual, expected)?;
    let mape = compute_mape(actual, expected)?;
    
    println!(
        "{}: max_diff={:.6e}, mean_diff={:.6e}, MAPE={:.2}%",
        name, max_diff, mean_diff, mape
    );
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_assert_tensor_close_passes() -> Result<()> {
        let device = Device::Cpu;
        
        let actual = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device)?;
        let expected = Tensor::from_vec(vec![1.0001f32, 2.0001, 3.0001], (3,), &device)?;
        
        assert_tensor_close(&actual, &expected, 1e-3, "test")?;
        
        Ok(())
    }

    #[test]
    fn test_assert_tensor_close_fails() -> Result<()> {
        let device = Device::Cpu;
        
        let actual = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device)?;
        let expected = Tensor::from_vec(vec![2.0f32, 3.0, 4.0], (3,), &device)?;
        
        let result = assert_tensor_close(&actual, &expected, 1e-3, "test");
        assert!(result.is_err());
        
        Ok(())
    }

    #[test]
    fn test_compute_max_diff() -> Result<()> {
        let device = Device::Cpu;
        
        let actual = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device)?;
        let expected = Tensor::from_vec(vec![1.1f32, 2.2, 2.8], (3,), &device)?;
        
        let max_diff = compute_max_diff(&actual, &expected)?;
        
        // Max diff should be 0.2
        assert!((max_diff - 0.2).abs() < 1e-5);
        
        Ok(())
    }
}
