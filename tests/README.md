# Correctness Testing Framework

This directory contains the correctness testing infrastructure for verifying the Sundial Rust implementation against Python reference outputs.

## Overview

The correctness testing framework enables automated verification that the Rust implementation produces results matching the Python reference implementation. This is critical for:

- **Bug Detection**: Catching implementation errors like the RoPE sign error
- **Regression Testing**: Ensuring fixes don't break anything
- **CI/CD Integration**: Automated validation on every build
- **Confidence**: Validating numerical correctness at each layer

## Architecture

```
tests/
├── reference_data/           # Python-generated reference tensors
│   ├── input.npy            # Test input
│   ├── patch_embed_output.npy
│   ├── layer_0_output.npy
│   ├── ...
│   ├── layer_11_output.npy
│   ├── transformer_output.npy
│   └── predictions.npy
├── correctness.rs           # End-to-end correctness test
├── layer_correctness.rs     # Layer-by-layer comparison
└── generate_reference.py    # Script to generate Python references
```

## Components

### 1. Python Reference Generation (`scripts/generate_reference.py`)

Generates reference tensors from the Python Sundial implementation:

```bash
# Generate full model intermediates
python scripts/generate_reference.py \
    --input test_data/input.npy \
    --output tests/reference_data/ \
    --full-model

# Generate RoPE-specific references
python scripts/generate_reference.py \
    --output tests/reference_data/rope/ \
    --rope-only
```

### 2. Rust Testing Infrastructure (`src/testing/`)

#### Reference Loader (`src/testing/reference_loader.rs`)

```rust
use sundial_rust::testing::load_reference_tensor;

// Load a single tensor
let tensor = load_reference_tensor("/path/to/tensor.npy")?;

// Load multiple tensors from a directory
let tensors = load_reference_tensors_from_dir("/path/to/dir")?;

// Load specific tensor by name
let tensor = load_tensor_by_name("/path/to/dir", "layer_0_output")?;
```

#### Assertions (`src/testing/assertions.rs`)

```rust
use sundial_rust::testing::{
    assert_tensor_close,
    assert_tensor_exact,
    assert_tensor_mape,
    compute_max_diff,
    compute_mean_diff,
    compute_mape,
    print_comparison_stats
};

// Assert tensors are close (custom tolerance)
assert_tensor_close(&actual, &expected, 1e-4, "Attention output")?;

// Assert tensors match exactly (tight tolerance: 1e-5)
assert_tensor_exact(&actual, &expected, "RoPE output")?;

// Assert within percentage tolerance
assert_tensor_mape(&actual, &expected, 5.0, "Predictions")?;

// Compute statistics
let max_diff = compute_max_diff(&actual, &expected)?;
let mean_diff = compute_mean_diff(&actual, &expected)?;
let mape = compute_mape(&actual, &expected)?;

// Print stats for debugging
print_comparison_stats(&actual, &expected, "Layer 5")?;
```

### 3. Component Tests

#### RoPE Correctness Test (`src/model/rope.rs`)

```rust
#[test]
fn test_rope_matches_python_reference() {
    use crate::testing::{load_tensor_by_name, assert_tensor_exact};
    
    let device = Device::Cpu;
    
    // Load Python reference
    let q = load_tensor_by_name("/tmp/rope_reference", "rope_q_input")?;
    let k = load_tensor_by_name("/tmp/rope_reference", "rope_k_input")?;
    let q_expected = load_tensor_by_name("/tmp/rope_reference", "rope_q_output")?;
    let k_expected = load_tensor_by_name("/tmp/rope_reference", "rope_k_output")?;
    
    // Run Rust implementation
    let rope = SundialRotaryEmbedding::new(64, 1000, 10000.0, &device)?;
    let (q_output, k_output) = rope.forward(&q, &k, None)?;
    
    // Assert against Python
    assert_tensor_exact(&q_output, &q_expected, "Q RoPE output")?;
    assert_tensor_exact(&k_output, &k_expected, "K RoPE output")?;
}
```

#### End-to-End Test (`tests/correctness.rs`)

```rust
#[test]
fn test_end_to_end_matches_python() {
    use sundial_rust::testing::{load_reference_tensor, assert_tensor_mape};
    
    let device = Device::Cpu;
    
    // Load test input and Python predictions
    let input = load_reference_tensor("tests/reference_data/input.npy")?;
    let python_predictions = load_reference_tensor("tests/reference_data/predictions.npy")?;
    
    // Load model and run inference
    let config = SundialConfig::default();
    let vb = load_sundial_from_huggingface("thuml/sundial-base-128m", &device)?;
    let model = SundialModel::new(&config, vb)?;
    
    let predictions = model.generate(&input, 14, 1, false)?;
    
    // Compare with Python (relaxed tolerance for full model)
    assert_tensor_mape(&predictions, &python_predictions, 5.0, "Final predictions")?;
}
```

#### Layer-by-Layer Test (`tests/layer_correctness.rs`)

```rust
#[test]
fn test_each_layer_matches_python() {
    use sundial_rust::testing::{load_reference_tensor, assert_tensor_close};
    
    let device = Device::Cpu;
    let config = SundialConfig::default();
    let vb = load_sundial_from_huggingface("thuml/sundial-base-128m", &device)?;
    let transformer = SundialTransformer::new(&config, vb)?;
    
    let input = load_reference_tensor("tests/reference_data/input.npy")?;
    
    // Patch embed
    let hidden = transformer.embed_layer.forward(&input)?;
    let expected = load_reference_tensor("tests/reference_data/patch_embed_output.npy")?;
    assert_tensor_close(&hidden, &expected, 1e-5, "After patch embed")?;
    
    // Each layer
    for layer_idx in 0..12 {
        let hidden = transformer.layers[layer_idx].forward(&hidden)?;
        let expected = load_reference_tensor(&format!(
            "tests/reference_data/layer_{}_output.npy",
            layer_idx
        ))?;
        
        // Tolerance increases with depth due to error accumulation
        let tolerance = 0.1 + (layer_idx as f32 * 0.05);
        assert_tensor_close(&hidden, &expected, tolerance, 
            &format!("Layer {}", layer_idx))?;
    }
}
```

## Tolerance Guidelines

Different components require different tolerances based on numerical stability:

| Component | Tolerance | Reason |
|-----------|-----------|--------|
| RoPE | 1e-5 | Pure math, should be nearly exact |
| Attention | 1e-4 | Softmax introduces small differences |
| Layer Norm | 1e-5 | Simple arithmetic, tight tolerance |
| MLP | 1e-5 | SiLU activation, tight tolerance |
| Patch Embed | 1e-5 | Linear layers, tight tolerance |
| Full Model | 5% MAPE | Error accumulation through layers |

## Running Tests

```bash
# Run all correctness tests
cargo test correctness

# Run specific test
cargo test test_rope_matches_python_reference

# Run layer-by-layer tests
cargo test --test layer_correctness

# Run end-to-end test
cargo test --test correctness
```

## CI/CD Integration

Add to `.github/workflows/correctness-tests.yml`:

```yaml
name: Correctness Tests

on: [push, pull_request]

jobs:
  correctness:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install Python dependencies
        run: |
          pip install torch numpy safetensors transformers
      
      - name: Generate Python reference
        run: |
          python scripts/generate_reference.py \
            --input tests/test_input.npy \
            --output /tmp/python_reference/
      
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Run correctness tests
        run: |
          cargo test --test correctness
          cargo test --test layer_correctness
```

## Test Data Management

### Generating Reference Data

```bash
# Generate RoPE references
python scripts/generate_reference.py \
    --output tests/reference_data/rope/ \
    --rope-only

# Generate full model references
python scripts/generate_reference.py \
    --input test_data/input.npy \
    --output tests/reference_data/ \
    --full-model
```

### Version Control

Reference tensors should be stored in Git LFS:

```bash
# Install Git LFS
git lfs install

# Add reference data
git lfs track "tests/reference_data/*.npy"
git add .gitattributes
git add tests/reference_data/
git commit -m "Add Python reference tensors"
```

## Debugging Tips

When tests fail:

1. **Check tensor shapes**: Ensure Rust and Python produce same shapes
2. **Inspect differences**: Use `print_comparison_stats()` to see max/mean diff
3. **Compare element-wise**: Save both tensors and compare in Python
4. **Check weight loading**: Verify weights match Python exactly
5. **Test in isolation**: Test each component separately

```rust
// Debug example
use sundial_rust::debug_utils::debug_tensor;

debug_tensor("Q input", &q);
debug_tensor("Q output", &q_output);
debug_tensor("Q expected", &q_expected);
```

## Lessons Learned

The RoPE sign error demonstrates why correctness tests are essential:

1. **Unit tests passed** - RoPE creation and forward worked with random data
2. **Bug was critical** - Caused negative correlation with Python
3. **Detection was manual** - Required extensive debugging
4. **Correctness test would catch it** - Comparing against Python reference would fail immediately

**Every component needs a correctness test against Python reference.**

## Next Steps

1. ✅ Create Python reference generation script
2. ✅ Implement Rust test utilities
3. ✅ Add RoPE correctness test
4. ⏳ Add attention correctness test
5. ⏳ Add MLP correctness test
6. ⏳ Add patch embedding correctness test
7. ⏳ Add end-to-end correctness test
8. ⏳ Add layer-by-layer tests
9. ⏳ Integrate with CI/CD
