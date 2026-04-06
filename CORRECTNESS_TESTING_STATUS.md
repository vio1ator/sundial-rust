# Correctness Testing Implementation Status

## Overview

This document tracks the implementation status of the correctness testing framework as outlined in PLAN.md.

## Implementation Summary

### ✅ Phase 1: Python Reference Generation Script

**Status**: COMPLETE

**Files Created**:
- `scripts/generate_reference.py` - Python script to generate reference tensors

**Features**:
- Generates full model intermediate tensors
- Generates RoPE-specific reference tensors
- Saves all intermediate layers (patch embed, 12 transformer layers, final norm)
- Saves metadata JSON with tensor information
- Supports both full model and RoPE-only generation modes

**Usage**:
```bash
# Generate full model references
python scripts/generate_reference.py \
    --input test_data/input.npy \
    --output tests/reference_data/ \
    --full-model

# Generate RoPE references only
python scripts/generate_reference.py \
    --output tests/reference_data/rope/ \
    --rope-only
```

---

### ✅ Phase 2: Rust Test Infrastructure

**Status**: COMPLETE

**Files Created**:
- `src/testing/mod.rs` - Testing module exports
- `src/testing/reference_loader.rs` - Load Python reference tensors from .npy files
- `src/testing/assertions.rs` - Tensor comparison assertions

**Features**:

#### Reference Loader
- `load_reference_tensor(path)` - Load single tensor from .npy file
- `load_reference_tensors_from_dir(dir)` - Load all tensors from directory
- `load_tensor_by_name(dir, name)` - Load specific tensor by name
- `save_tensor_to_npy(tensor, path)` - Save tensor for debugging

#### Assertions
- `assert_tensor_close(actual, expected, tolerance, name)` - Assert within tolerance
- `assert_tensor_exact(actual, expected, name)` - Assert with tight tolerance (1e-5)
- `assert_tensor_relaxed(actual, expected, name)` - Assert with relaxed tolerance
- `assert_tensor_mape(actual, expected, max_mape, name)` - Assert within percentage
- `compute_max_diff(actual, expected)` - Compute maximum absolute difference
- `compute_mean_diff(actual, expected)` - Compute mean absolute difference
- `compute_mape(actual, expected)` - Compute mean absolute percentage error
- `print_comparison_stats(actual, expected, name)` - Print debug statistics

**Dependencies Added**:
- `ndarray = "0.16"` - For .npy file handling
- `ndarray-npy = "0.9"` - For reading/writing .npy files

---

### ✅ Phase 3: Component-Level Correctness Tests

**Status**: PARTIAL - RoPE test implemented

**Files Modified**:
- `src/model/rope.rs` - Added `test_rope_matches_python_reference()` test

**Test Coverage**:
- ✅ RoPE (Rotary Positional Embeddings)
- ⏳ Attention (pending Python hooks)
- ⏳ MLP (pending Python hooks)
- ⏳ Patch Embedding (pending Python hooks)

**RoPE Test Example**:
```rust
#[test]
fn test_rope_matches_python_reference() {
    use crate::testing::{load_tensor_by_name, assert_tensor_exact};
    
    let device = Device::Cpu;
    
    // Load Python reference inputs and outputs
    let q = load_tensor_by_name("/tmp/rope_reference", "rope_q_input")?;
    let k = load_tensor_by_name("/tmp/rope_reference", "rope_k_input")?;
    let q_expected = load_tensor_by_name("/tmp/rope_reference", "rope_q_output")?;
    let k_expected = load_tensor_by_name("/tmp/rope_reference", "rope_k_output")?;
    
    // Create RoPE and run
    let rope = SundialRotaryEmbedding::new(64, 1000, 10000.0, &device)?;
    let (q_output, k_output) = rope.forward(&q, &k, None)?;
    
    // Assert against Python reference
    assert_tensor_exact(&q_output, &q_expected, "Q RoPE output")?;
    assert_tensor_exact(&k_output, &k_expected, "K RoPE output")?;
}
```

---

### ⏳ Phase 4: End-to-End Correctness Test

**Status**: IMPLEMENTED BUT PENDING REFERENCE DATA

**Files Created**:
- `tests/correctness.rs` - End-to-end correctness test suite

**Tests**:
- ✅ `test_end_to_end_matches_python()` - Full model output comparison
- ✅ `test_end_to_end_various_sizes()` - Test with different input sizes

**Requirements**:
- Generate Python reference predictions
- Store in `tests/reference_data/predictions.npy`

---

### ⏳ Phase 5: Layer-by-Layer Correctness Test

**Status**: IMPLEMENTED BUT PENDING REFERENCE DATA

**Files Created**:
- `tests/layer_correctness.rs` - Layer-by-layer comparison tests

**Tests**:
- ✅ `test_each_layer_matches_python()` - Verify all 12 transformer layers
- ⏳ `test_layer_components_match_python()` - Individual component tests (needs Python hooks)

**Requirements**:
- Generate Python reference for all intermediate layers
- Store in `tests/reference_data/layer_{0-11}_output.npy`

---

### ⏳ Phase 6: CI/CD Integration

**Status**: NOT STARTED

**To Do**:
- Create `.github/workflows/correctness-tests.yml`
- Set up Git LFS for reference data
- Configure automated reference generation in CI

---

## Test Utilities Documentation

### Tolerance Guidelines

| Component | Tolerance | Reason |
|-----------|-----------|--------|
| RoPE | 1e-5 | Pure math, should be nearly exact |
| Attention | 1e-4 | Softmax introduces small differences |
| Layer Norm | 1e-5 | Simple arithmetic, tight tolerance |
| MLP | 1e-5 | SiLU activation, tight tolerance |
| Patch Embed | 1e-5 | Linear layers, tight tolerance |
| Full Model | 5% MAPE | Error accumulation through layers |

### Usage Examples

```rust
use sundial_rust::testing::*;

// Load reference tensor
let expected = load_reference_tensor("path/to/tensor.npy")?;

// Assert exact match (1e-5 tolerance)
assert_tensor_exact(&actual, &expected, "Component name")?;

// Assert with custom tolerance
assert_tensor_close(&actual, &expected, 1e-4, "Component name")?;

// Assert within percentage
assert_tensor_mape(&actual, &expected, 5.0, "Component name")?;

// Compute statistics
let max_diff = compute_max_diff(&actual, &expected)?;
let mean_diff = compute_mean_diff(&actual, &expected)?;
let mape = compute_mape(&actual, &expected)?;

// Print for debugging
print_comparison_stats(&actual, &expected, "Component name")?;
```

---

## Running Tests

```bash
# Run all testing module unit tests
cargo test --lib testing

# Run RoPE correctness test
cargo test test_rope_matches_python_reference

# Run end-to-end test (requires reference data)
cargo test --test correctness

# Run layer-by-layer tests (requires reference data)
cargo test --test layer_correctness
```

---

## Next Steps

### Immediate (Critical)

1. **Generate Python Reference Data**
   - Run `python scripts/generate_reference.py` to create reference tensors
   - Store in `tests/reference_data/`
   - Add to Git LFS

2. **Test RoPE Correctness**
   - Generate RoPE-specific references
   - Run `cargo test test_rope_matches_python_reference`
   - Verify RoPE implementation matches Python

### Short Term (High Priority)

3. **Add Attention Correctness Test**
   - Modify `src/model/attention.rs` to add test
   - Generate attention-specific Python references
   - Verify attention implementation

4. **Add MLP Correctness Test**
   - Modify `src/model/mlp.rs` to add test
   - Generate MLP-specific Python references
   - Verify MLP implementation

5. **Add Patch Embedding Correctness Test**
   - Modify `src/model/patch_embed.rs` to add test
   - Generate patch embed Python references
   - Verify patch embedding implementation

### Medium Term

6. **End-to-End Test Verification**
   - Generate full model Python predictions
   - Run end-to-end test
   - Verify overall model correctness

7. **Layer-by-Layer Test Verification**
   - Generate all layer intermediates
   - Run layer-by-layer test
   - Identify any layer-specific issues

### Long Term

8. **CI/CD Integration**
   - Create GitHub Actions workflow
   - Set up automated reference generation
   - Configure Git LFS for large files

9. **Documentation**
   - Complete testing README
   - Add examples for common scenarios
   - Document debugging techniques

---

## Lessons from RoPE Bug

The RoPE sign error we fixed demonstrates why this framework is essential:

1. **Unit tests passed** - RoPE creation and forward worked with random data
2. **Bug was critical** - Caused negative correlation with Python
3. **Detection was manual** - Required extensive debugging to find
4. **Correctness test would catch it** - Comparing against Python reference would fail immediately

**Key Takeaway**: Every component needs a correctness test against Python reference.

---

## Files Modified/Created

### Created
- `scripts/generate_reference.py`
- `src/testing/mod.rs`
- `src/testing/reference_loader.rs`
- `src/testing/assertions.rs`
- `tests/correctness.rs`
- `tests/layer_correctness.rs`
- `tests/README.md`
- `CORRECTNESS_TESTING_STATUS.md`

### Modified
- `Cargo.toml` - Added `ndarray` and `ndarray-npy` dependencies
- `src/lib.rs` - Added `testing` module
- `src/model/rope.rs` - Added correctness test

---

## Verification

Build and test the infrastructure:

```bash
# Verify build
cargo check

# Run testing module unit tests
cargo test --lib testing

# Run RoPE test (will fail until reference data is generated)
cargo test test_rope_matches_python_reference
```

All infrastructure is in place and working. The framework is ready for use once Python reference data is generated.
