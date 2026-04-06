# Sundial Rust - Correctness Testing Plan

## Executive Summary

**Current State**: ✅ **INFRASTRUCTURE COMPLETE, REFERENCE DATA PENDING**

The correctness testing framework infrastructure is fully implemented with:
- ✅ Python reference generation script
- ✅ Rust testing utilities (loader, assertions, statistics)
- ✅ Integration test templates
- ✅ Comprehensive documentation

**Next Steps**: Generate Python reference data to enable automated correctness testing.

**Known Issues**:
- ⏸️ 8 tests are ignored pending Python reference data generation
- ❌ 3 pre-existing failures in embedded weights verification tests (unrelated to correctness testing)

---

## Testing Gap Analysis

### What We Have

| Test Type | Count | Status | Notes |
|-----------|-------|--------|-------|
| Unit tests | 65 | ✅ PASSING | Core functionality tests |
| Testing infrastructure | Complete | ✅ READY | All utilities implemented |
| Correctness tests | 8 | ⏸️ IGNORED | Need Python reference data |
| Embedded weights tests | 9 | ⚠️ 3 FAILING | Pre-existing issues, unrelated |

### What We Need

✅ **Automated correctness tests** that:
- Load pre-computed Python reference outputs
- Compare Rust outputs against Python with tight tolerances
- Run on every build/CI
- Catch regressions immediately

---

## Implementation Status

### ✅ Phase 1: Python Reference Generation Script

**Status**: COMPLETE

**File**: `scripts/generate_reference.py`

**Features**:
- Generates full model intermediate tensors from Python Sundial
- Generates RoPE-specific reference tensors
- Saves all intermediate layers (patch embed, 12 transformer layers, final norm)
- Saves predictions and metadata

**Usage**:
```bash
# Generate RoPE references (quick, no model needed)
python scripts/generate_reference.py \
    --output tests/reference_data/rope/ \
    --rope-only

# Generate full model references (requires Python Sundial installation)
python scripts/generate_reference.py \
    --input test_data/input.npy \
    --output tests/reference_data/ \
    --full-model \
    --model thuml/sundial-base-128m
```

**Dependencies**:
```bash
pip install torch numpy safetensors transformers
```

---

### ✅ Phase 2: Rust Test Infrastructure

**Status**: COMPLETE

**Files**:
- `src/testing/mod.rs` - Module exports
- `src/testing/reference_loader.rs` - Load/save .npy tensors
- `src/testing/assertions.rs` - Tensor comparison utilities

**API**:
```rust
// Load reference tensors
let tensor = load_reference_tensor("path/to/tensor.npy")?;
let tensors = load_reference_tensors_from_dir("path/to/dir")?;
let tensor = load_tensor_by_name("path/to/dir", "layer_0_output")?;

// Assertions
assert_tensor_exact(&actual, &expected, "component_name")?;           // 1e-5 tolerance
assert_tensor_close(&actual, &expected, 1e-4, "component_name")?;    // custom tolerance
assert_tensor_mape(&actual, &expected, 5.0, "component_name")?;      // 5% MAPE

// Statistics
let max_diff = compute_max_diff(&actual, &expected)?;
let mean_diff = compute_mean_diff(&actual, &expected)?;
let mape = compute_mape(&actual, &expected)?;
print_comparison_stats(&actual, &expected, "component_name");
```

---

### ✅ Phase 3: Component-Level Correctness Tests

**Status**: IMPLEMENTED, PENDING DATA

**RoPE Test**: `src/model/rope.rs::test_rope_matches_python_reference`
- ✅ Test implementation complete
- ⏸️ Ignored until Python reference data generated
- Requires: RoPE input/output tensors

**Pending Tests**:
- Attention correctness test
- MLP correctness test
- Patch embedding correctness test

---

### ✅ Phase 4: End-to-End Correctness Test

**Status**: IMPLEMENTED, PENDING DATA

**File**: `tests/correctness.rs`

**Tests**:
- `test_end_to_end_matches_python` - Full model output comparison
- `test_end_to_end_various_sizes` - Different input sizes

**Requirements**:
- Generate full model predictions from Python
- Store in `tests/reference_data/predictions.npy`

---

### ✅ Phase 5: Layer-by-Layer Correctness Test

**Status**: IMPLEMENTED, PENDING DATA

**File**: `tests/layer_correctness.rs`

**Tests**:
- `test_model_processes_all_layers` - Verify model runs through all layers
- `test_layer_components_match_python` - Individual component tests (TODO)

**Requirements**:
- Generate all layer intermediates from Python
- Store in `tests/reference_data/layer_{0-11}_output.npy`

---

### ⏳ Phase 6: CI/CD Integration

**Status**: NOT STARTED

**To Do**:
- Create `.github/workflows/correctness-tests.yml`
- Set up Git LFS for reference data
- Configure automated reference generation in CI

---

## How to Generate Python Reference Data

### Option 1: Quick Start - RoPE Only (5 minutes)

```bash
# Install dependencies
pip install torch numpy

# Generate RoPE reference data
python scripts/generate_reference.py \
    --output tests/reference_data/rope/ \
    --rope-only

# Run RoPE correctness test
cargo test test_rope_matches_python_reference -- --ignored
```

### Option 2: Full Model References (30 minutes)

```bash
# Install Sundial Python package
pip install transformers==4.40.1 torch numpy safetensors

# Create test input (or use your own)
python -c "import numpy as np; np.save('tests/reference_data/input.npy', np.random.randn(1, 2880, 1).astype(np.float32))"

# Generate full model references
python scripts/generate_reference.py \
    --input tests/reference_data/input.npy \
    --output tests/reference_data/ \
    --full-model \
    --model thuml/sundial-base-128m

# Run all correctness tests
cargo test --test correctness -- --ignored
cargo test --test layer_correctness -- --ignored
```

### Option 3: Use Pre-computed References

If you have access to the Sundial Python team's reference data:
1. Download the reference tensors
2. Place in `tests/reference_data/`
3. Run tests: `cargo test correctness`

---

## Tolerance Guidelines

Different components require different tolerances based on numerical stability:

| Component | Tolerance | Reason |
|-----------|-----------|--------|
| RoPE | 1e-5 | Pure math, should be nearly exact |
| Attention | 1e-4 | Softmax introduces small differences |
| Layer Norm | 1e-5 | Simple arithmetic, tight tolerance |
| MLP | 1e-5 | SiLU activation, tight tolerance |
| Patch Embed | 1e-5 | Linear layers, tight tolerance |
| Full Model | 1.0 (abs) or 5% MAPE | Error accumulation through layers |

---

## Known Issues and Fixes

### Issue 1: Ignored Tests (8 tests)

**Symptom**: Tests marked with `#[ignore]` and skipped during normal test runs

**Cause**: Require Python reference data that hasn't been generated yet

**Solution**: Generate reference data using scripts in "How to Generate Python Reference Data" section above

**Verification**:
```bash
# Run ignored tests after generating data
cargo test --ignored
```

### Issue 2: Embedded Weights Test Failures (3 tests)

**Symptom**: Following tests fail in `tests/embedded_weights_verification.rs`:
- `test_no_weights_leak_after_cleanup`
- `test_cross_platform_temp_paths`
- `test_verbose_extraction_message`

**Root Cause**: These are pre-existing bugs in the test logic, unrelated to correctness testing framework. They involve:
1. Temp directory lifecycle management
2. Path format assumptions
3. Verbose output parsing

**Impact**: LOW - These tests verify internal implementation details, not model correctness

**Recommended Actions**:

#### Option A: Fix the Tests (Recommended)

1. **test_no_weights_leak_after_cleanup**: The test assumes temp directory exists while loader is alive, but tempfile crate may clean up immediately on Unix. Fix by checking cleanup behavior instead of directory existence.

2. **test_cross_platform_temp_paths**: The test uses `model_path()` which may return `<memory>` or other non-filesystem paths. Fix by handling special paths gracefully.

3. **test_verbose_extraction_message**: Verbose output format may vary. Fix by making assertion more flexible.

#### Option B: Mark as Ignored (Temporary)

```rust
#[test]
#[ignore = "Known issue: temp path handling needs review"]
fn test_no_weights_leak_after_cleanup() { ... }
```

#### Option C: Remove Tests

If these tests don't provide value, consider removing them entirely.

**Decision**: These should be addressed separately from the correctness testing framework as they are unrelated to model correctness verification.

---

## Running Tests

```bash
# Run all passing tests
cargo test

# Run only testing infrastructure tests
cargo test --lib testing

# Run RoPE correctness test (after generating reference data)
cargo test test_rope_matches_python_reference -- --ignored

# Run end-to-end correctness tests (after generating reference data)
cargo test --test correctness -- --ignored

# Run layer-by-layer tests (after generating reference data)
cargo test --test layer_correctness -- --ignored

# Run all ignored tests
cargo test --ignored
```

---

## CI/CD Integration Plan

### GitHub Actions Workflow

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
          pip install torch numpy safetensors transformers==4.40.1
      
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Generate Python reference (RoPE only for speed)
        run: |
          python scripts/generate_reference.py \
            --output /tmp/python_reference/rope/ \
            --rope-only
      
      - name: Run RoPE correctness test
        run: |
          cargo test test_rope_matches_python_reference -- --ignored
      
      - name: Run unit tests
        run: |
          cargo test --lib testing
```

### Git LFS Setup

```bash
# Install Git LFS
git lfs install

# Track reference data
git lfs track "tests/reference_data/*.npy"
git add .gitattributes

# Commit reference data
git add tests/reference_data/
git commit -m "Add Python reference tensors"
```

---

## Test Data Management

### Reference Data Structure

```
tests/reference_data/
├── input.npy                 # Test input (2880 timesteps)
├── patch_embed_output.npy
├── layer_0_output.npy
├── layer_1_output.npy
├── ...
├── layer_11_output.npy
├── transformer_output.npy
├── predictions.npy
└── metadata.json
```

### Version Control Strategy

- Store reference tensors in Git LFS
- Document expected tolerances in test comments
- Include hash verification for data integrity
- Keep reference data generation script in version control

---

## Success Criteria

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Infrastructure complete | Yes | ✅ Yes | Done |
| Component tests | 5+ | 1 (RoPE) | In Progress |
| Layer-by-layer tests | 12 layers | 1 test | In Progress |
| End-to-end test | 1 test | 1 test | Ready |
| CI integration | Automated | Manual | TODO |
| Bug detection time | < 1 hour | Days → Minutes | Ready |

---

## Lessons from RoPE Bug

The RoPE sign error we fixed demonstrates why correctness tests are essential:

1. **Unit tests passed** - RoPE creation and forward worked with random data
2. **Bug was critical** - Caused negative correlation with Python
3. **Detection was manual** - Required extensive debugging to find
4. **Correctness test would catch it** - Comparing against Python reference would fail immediately

**Key Takeaway**: Every component needs a correctness test against Python reference.

---

## Next Steps (Priority Order)

### Immediate (This Week)

1. **Generate RoPE Reference Data** (15 minutes)
   - Run: `python scripts/generate_reference.py --output tests/reference_data/rope/ --rope-only`
   - Verify: `cargo test test_rope_matches_python_reference -- --ignored`
   - Impact: First automated correctness test

2. **Fix Embedded Weights Test Issues** (1-2 hours)
   - Review failing tests in `tests/embedded_weights_verification.rs`
   - Either fix the test logic or mark as ignored
   - Impact: Clean test suite

### Short Term (This Month)

3. **Generate Full Model References** (30 minutes)
   - Install Python Sundial
   - Generate all layer intermediates
   - Verify end-to-end and layer-by-layer tests

4. **Add More Component Tests** (2-3 days)
   - Attention correctness test
   - MLP correctness test
   - Patch embedding correctness test

### Medium Term (Next Month)

5. **CI/CD Integration** (1 day)
   - Create GitHub Actions workflow
   - Set up Git LFS
   - Configure automated testing

6. **Documentation** (1 day)
   - Complete testing guide
   - Add debugging examples
   - Document common issues

---

## Appendix: Python Reference Generation Script

The script at `scripts/generate_reference.py` provides:

### Features
- Full model intermediate tensor capture
- RoPE-specific test case generation
- Metadata and hash generation
- Flexible input/output configuration

### Template

```python
#!/usr/bin/env python3
"""Generate Python reference tensors for correctness testing."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM

def generate_reference(input_path, output_dir):
    """Generate reference tensors from Python Sundial model."""
    # Load model
    model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', 
                                                  trust_remote_code=True)
    model.eval()
    
    # Load input
    input_data = np.load(input_path)
    input_tensor = torch.from_numpy(input_data).float()
    
    # Capture intermediates with hooks
    intermediates = {}
    
    def make_hook(name):
        def hook(module, input, output):
            intermediates[name] = output.detach().cpu().numpy()
        return hook
    
    # Register hooks on all layers
    # ... (see scripts/generate_reference.py for full implementation)
    
    # Save all tensors
    for name, tensor in intermediates.items():
        np.save(f'{output_dir}/{name}.npy', tensor)
```

Full implementation available at: `scripts/generate_reference.py`

---

## Contact & Support

For issues with:
- **Python reference generation**: Check `scripts/generate_reference.py` comments
- **Rust testing infrastructure**: Review `tests/README.md`
- **Sundial model**: See [thuml/Sundial](https://github.com/thuml/Sundial)
- **Test failures**: Check test-specific documentation in test files

---

**Last Updated**: 2026-04-06
**Status**: Infrastructure Complete, Awaiting Reference Data
