# Sundial Rust - Correctness Testing Plan

## Executive Summary

**Current State**: ❌ **NO ACTUAL CORRECTNESS TESTS**

While the project has 65 unit tests, **none of them verify the Rust implementation against the Python reference**. All tests use random data or mock inputs, which cannot catch implementation bugs like the RoPE sign error we just fixed.

**Impact**: 
- Critical bugs can go undetected until manual comparison
- No automated regression testing
- Cannot verify fixes without manual Python comparison
- High risk of silent numerical discrepancies

---

## Testing Gap Analysis

### What We Have

| Test Type | Count | Verifies Against Python? |
|-----------|-------|-------------------------|
| Unit tests | 65 | ❌ No - uses random data |
| Weight loading tests | 15 | ❌ No - only checks loading works |
| Component tests | 20+ | ❌ No - uses synthetic inputs |
| Comparison examples | 5 | ⚠️ Manual only, not automated |

### What We Need

✅ **Automated correctness tests** that:
- Load pre-computed Python reference outputs
- Compare Rust outputs against Python with tight tolerances
- Run on every build/CI
- Catch regressions immediately

---

## Testing Strategy

### Phase 1: Create Python Reference Dataset (Priority: CRITICAL)

**Goal**: Generate golden reference tensors from Python implementation

**Steps**:

1. **Create Python Hook Script**
   ```python
   # scripts/generate_reference.py
   # Hook into Python model at each layer
   # Save all intermediate tensors to disk
   ```

2. **Generate Reference Data**
   ```bash
   python scripts/generate_reference.py \
       --input /tmp/test_input.npy \
       --output /tmp/python_reference/ \
       --save-intermediates
   ```

3. **Reference Data Structure**
   ```
   /tmp/python_reference/
   ├── input.npy                 # Input tensor
   ├── patch_embed_output.npy    # After patch embedding
   ├── layer_0_output.npy        # After decoder layer 0
   ├── layer_1_output.npy        # After decoder layer 1
   ...
   ├── layer_11_output.npy       # After decoder layer 11
   ├── transformer_output.npy    # Final transformer output
   └── predictions.npy           # Final predictions
   ```

**Success Criteria**:
- All intermediate tensors saved
- Tensors match shapes expected by Rust
- Documentation of what each tensor represents

---

### Phase 2: Add Rust Test Infrastructure (Priority: CRITICAL)

**Goal**: Create test utilities to load and compare Python references

**Files to Create**:

1. `src/testing/mod.rs` - Test utilities module
2. `src/testing/reference_loader.rs` - Load Python reference tensors
3. `src/testing/assertions.rs` - Tensor comparison assertions

**Implementation**:

```rust
// src/testing/reference_loader.rs
pub fn load_reference_tensor(path: &str) -> Result<Tensor> {
    // Load .npy or .bin file
    // Return tensor for comparison
}

// src/testing/assertions.rs
pub fn assert_tensor_close(
    actual: &Tensor,
    expected: &Tensor,
    tolerance: f32,
    name: &str,
) -> Result<()> {
    let diff = (actual - expected)?.abs()?;
    let max_diff = diff.max()?.to_scalar::<f32>()?;
    let mean_diff = diff.mean()?.to_scalar::<f32>()?;
    
    if max_diff > tolerance {
        bail!(
            "{}: max_diff={:.6e} > tolerance={:.6e}, mean_diff={:.6e}",
            name, max_diff, tolerance, mean_diff
        );
    }
    
    Ok(())
}

pub fn assert_tensor_exact(
    actual: &Tensor,
    expected: &Tensor,
    name: &str,
) -> Result<()> {
    assert_tensor_close(actual, expected, 1e-6, name)
}
```

---

### Phase 3: Add Component-Level Correctness Tests (Priority: HIGH)

**Goal**: Test individual components against Python references

#### 3.1 RoPE Correctness Test

**File**: `src/model/rope.rs` - Add test module

```rust
#[cfg(test)]
mod correctness_tests {
    use super::*;
    use crate::testing::{load_reference_tensor, assert_tensor_close};

    #[test]
    fn test_rope_matches_python_reference() {
        let device = Device::Cpu;
        
        // Load Python reference inputs and outputs
        let q = load_reference_tensor("/tmp/python_reference/rope_q_input.npy").unwrap();
        let k = load_reference_tensor("/tmp/python_reference/rope_k_input.npy").unwrap();
        let q_expected = load_reference_tensor("/tmp/python_reference/rope_q_output.npy").unwrap();
        let k_expected = load_reference_tensor("/tmp/python_reference/rope_k_output.npy").unwrap();
        
        // Create RoPE and run
        let rope = SundialRotaryEmbedding::new(64, 1000, 10000.0, &device).unwrap();
        let (q_output, k_output) = rope.forward(&q, &k, None).unwrap();
        
        // Assert against Python reference
        assert_tensor_close(&q_output, &q_expected, 1e-5, "Q RoPE output").unwrap();
        assert_tensor_close(&k_output, &k_expected, 1e-5, "K RoPE output").unwrap();
    }
}
```

#### 3.2 Attention Correctness Test

**File**: `src/model/attention.rs` - Add test module

```rust
#[cfg(test)]
mod correctness_tests {
    use super::*;
    use crate::testing::{load_reference_tensor, assert_tensor_close};

    #[test]
    fn test_attention_matches_python_reference() {
        // Load reference tensors
        let hidden = load_reference_tensor("/tmp/python_reference/attention_input.npy").unwrap();
        let expected = load_reference_tensor("/tmp/python_reference/attention_output.npy").unwrap();
        
        // Create attention layer with real weights (not random)
        let config = AttentionConfig {
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            attention_dropout: 0.0,
            max_position_embeddings: 10000,
            rope_theta: 10000.0,
            layer_idx: Some(0),
        };
        
        // Use real weights from Python reference
        let vb = create_varbuilder_from_python("/tmp/python_reference/layer_0/");
        let attn = SundialAttention::new(&config, vb).unwrap();
        
        // Run and compare
        let output = attn.forward(&hidden).unwrap();
        assert_tensor_close(&output, &expected, 1e-4, "Attention output").unwrap();
    }
}
```

#### 3.3 MLP Correctness Test

**File**: `src/model/mlp.rs` - Add test module

```rust
#[cfg(test)]
mod correctness_tests {
    use super::*;
    use crate::testing::{load_reference_tensor, assert_tensor_close};

    #[test]
    fn test_mlp_matches_python_reference() {
        let hidden = load_reference_tensor("/tmp/python_reference/mlp_input.npy").unwrap();
        let expected = load_reference_tensor("/tmp/python_reference/mlp_output.npy").unwrap();
        
        let config = SundialConfig::default();
        let vb = create_varbuilder_from_python("/tmp/python_reference/layer_0/mlp/");
        let mlp = SundialMLP::new(
            config.hidden_size,
            config.intermediate_size,
            &config.hidden_act,
            vb,
        ).unwrap();
        
        let output = mlp.forward(&hidden).unwrap();
        assert_tensor_close(&output, &expected, 1e-5, "MLP output").unwrap();
    }
}
```

#### 3.4 Patch Embedding Correctness Test

**File**: `src/model/patch_embed.rs` - Add test module

```rust
#[cfg(test)]
mod correctness_tests {
    use super::*;
    use crate::testing::{load_reference_tensor, assert_tensor_close};

    #[test]
    fn test_patch_embed_matches_python_reference() {
        let input = load_reference_tensor("/tmp/python_reference/input.npy").unwrap();
        let expected = load_reference_tensor("/tmp/python_reference/patch_embed_output.npy").unwrap();
        
        let config = SundialConfig::default();
        let vb = create_varbuilder_from_python("/tmp/python_reference/embed_layer/");
        let patch_embed = SundialPatchEmbedding::new(&config, vb).unwrap();
        
        let output = patch_embed.forward(&input).unwrap();
        assert_tensor_close(&output, &expected, 1e-5, "Patch embed output").unwrap();
    }
}
```

---

### Phase 4: Add End-to-End Correctness Test (Priority: HIGH)

**Goal**: Verify full model output matches Python

**File**: `tests/correctness.rs`

```rust
use candle_core::{Device, Tensor};
use sundial_rust::{SundialConfig, SundialModel};
use sundial_rust::testing::{load_reference_tensor, assert_tensor_close};

#[test]
fn test_end_to_end_matches_python() {
    let device = Device::Cpu;
    
    // Load test input
    let input = load_reference_tensor("/tmp/python_reference/input.npy").unwrap();
    
    // Load Python predictions
    let python_predictions = load_reference_tensor("/tmp/python_reference/predictions.npy").unwrap();
    
    // Load model with real weights
    let config = SundialConfig::default();
    let vb = sundial_rust::model::loader::load_sundial_from_huggingface(
        "thuml/sundial-base-128m",
        &device,
    ).unwrap();
    
    let model = SundialModel::new(&config, vb).unwrap();
    
    // Run inference
    let predictions = model.generate(&input, 14, 1, false).unwrap();
    
    // Compare with Python
    assert_tensor_close(&predictions, &python_predictions, 1.0, "Final predictions")
        .expect("Predictions should match Python reference");
}
```

---

### Phase 5: Add Layer-by-Layer Correctness Test (Priority: MEDIUM)

**Goal**: Verify each transformer layer matches Python

**File**: `tests/layer_correctness.rs`

```rust
use candle_core::{Device, Tensor};
use sundial_rust::{SundialConfig, SundialModel, SundialTransformer};
use sundial_rust::testing::{load_reference_tensor, assert_tensor_close};

#[test]
fn test_each_layer_matches_python() {
    let device = Device::Cpu;
    let config = SundialConfig::default();
    let vb = sundial_rust::model::loader::load_sundial_from_huggingface(
        "thuml/sundial-base-128m",
        &device,
    ).unwrap();
    
    let transformer = SundialTransformer::new(&config, vb).unwrap();
    
    // Load input
    let input = load_reference_tensor("/tmp/python_reference/input.npy").unwrap();
    
    // Patch embed
    let hidden = transformer.embed_layer.forward(&input).unwrap();
    let expected = load_reference_tensor("/tmp/python_reference/patch_embed_output.npy").unwrap();
    assert_tensor_close(&hidden, &expected, 1e-5, "After patch embed").unwrap();
    
    // Each layer
    for layer_idx in 0..12 {
        let hidden = transformer.layers[layer_idx].forward(&hidden).unwrap();
        let expected = load_reference_tensor(&format!(
            "/tmp/python_reference/layer_{}_output.npy",
            layer_idx
        )).unwrap();
        
        let max_diff = compute_max_diff(&hidden, &expected);
        println!("Layer {}: max_diff = {:.6e}", layer_idx, max_diff);
        
        // Tolerance increases slightly with depth due to error accumulation
        let tolerance = 0.1 + (layer_idx as f32 * 0.05);
        assert_tensor_close(&hidden, &expected, tolerance, 
            &format!("Layer {}", layer_idx)).unwrap();
    }
}
```

---

### Phase 6: CI/CD Integration (Priority: MEDIUM)

**Goal**: Automate correctness tests in CI pipeline

**GitHub Actions Workflow** (`.github/workflows/correctness-tests.yml`):

```yaml
name: Correctness Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

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
          pip install torch safetensors transformers
      
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

---

## Implementation Timeline

| Phase | Description | Duration | Dependencies |
|-------|-------------|----------|--------------|
| 1 | Create Python reference dataset | 1 day | None |
| 2 | Add Rust test infrastructure | 1 day | None |
| 3 | Component-level correctness tests | 2-3 days | Phases 1-2 |
| 4 | End-to-end correctness test | 1 day | Phases 1-3 |
| 5 | Layer-by-layer tests | 1-2 days | Phases 1-4 |
| 6 | CI/CD integration | 1 day | All above |
| **Total** | | **7-9 days** | |

---

## Test Tolerance Guidelines

Different components require different tolerances:

| Component | Tolerance | Reason |
|-----------|-----------|--------|
| RoPE | 1e-5 | Pure math, should be nearly exact |
| Attention | 1e-4 | Softmax introduces small numerical differences |
| Layer Norm | 1e-5 | Simple arithmetic, tight tolerance |
| MLP | 1e-5 | SiLU activation, tight tolerance |
| Patch Embed | 1e-5 | Linear layers, tight tolerance |
| Full Model | 1.0 | Error accumulation through 12 layers |
| End-to-End | 5% MAPE | Acceptable for production |

---

## Test Data Management

### Reference Data Storage

```
tests/
├── reference_data/
│   ├── input.npy                 # Test input (2880 timesteps)
│   ├── patch_embed_output.npy
│   ├── layer_0_*.npy             # All 12 layers
│   ├── transformer_output.npy
│   └── predictions.npy
├── test_input.npy                # Raw test data
└── generate_reference.py         # Script to regenerate references
```

### Version Control Strategy

- Store reference tensors in Git LFS
- Document expected tolerances in README
- Include hash verification for reference data integrity

---

## Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| Component tests against Python | 5+ | 0 |
| Layer-by-layer tests | 12 layers | 0 |
| End-to-end test | 1 test | 0 |
| CI integration | Automated | Manual |
| Bug detection time | < 1 hour | Days/weeks |

---

## Immediate Next Steps

1. **Create Python reference generation script** (Day 1)
   - Hook into Python model
   - Save all intermediate tensors
   - Verify tensor shapes

2. **Implement Rust test utilities** (Day 1-2)
   - Tensor loading from .npy files
   - Comparison assertions with tolerances
   - Error reporting utilities

3. **Add RoPE correctness test** (Day 2-3)
   - First component test
   - Validate against Python reference
   - Catch the sign error we just fixed

4. **Run and iterate** (Day 3+)
   - Identify remaining bugs
   - Fix issues
   - Add more component tests

---

## Lessons from RoPE Bug

The RoPE sign error we just fixed demonstrates why correctness tests are essential:

1. **Unit tests passed** - RoPE creation and forward worked with random data
2. **Bug was critical** - Caused negative correlation with Python
3. **Detection was manual** - Required extensive debugging to find
4. **Correctness test would catch it** - Comparing against Python reference would fail immediately

**Every component needs a correctness test against Python reference.**

---

## Appendix: Python Reference Generation Script Template

```python
# scripts/generate_reference.py
import torch
import numpy as np
from safetensors import safe_open

def generate_reference(input_path, output_dir):
    """Generate Python reference tensors for all intermediate layers"""
    
    # Load model
    model = load_sundial_model()
    model.eval()
    
    # Load input
    input_data = np.load(input_path)
    input_tensor = torch.from_numpy(input_data).float()
    
    # Create hooks to capture intermediate outputs
    intermediates = {}
    
    def make_hook(name):
        def hook(module, input, output):
            intermediates[name] = output.detach().cpu().numpy()
        return hook
    
    # Register hooks on all layers
    model.model.embed_layer.register_forward_hook(make_hook('patch_embed_output'))
    for i in range(12):
        model.model.layers[i].register_forward_hook(make_hook(f'layer_{i}_output'))
    model.model.norm.register_forward_hook(make_hook('transformer_output'))
    
    # Run forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # Save predictions
    intermediates['predictions'] = output.detach().cpu().numpy()
    
    # Save all intermediates
    os.makedirs(output_dir, exist_ok=True)
    for name, tensor in intermediates.items():
        np.save(f'{output_dir}/{name}.npy', tensor)
        print(f"Saved {name}: shape={tensor.shape}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    generate_reference(args.input, args.output)
```
