# Sundial Rust Debugging Guide

This guide explains how to debug the Sundial Rust implementation by comparing it with the Python reference.

## Overview

The debugging workflow involves:
1. Running both Python and Rust implementations with the same input
2. Capturing intermediate tensors from both
3. Comparing them to identify where deviations occur

## Prerequisites

### Python Dependencies
```bash
pip install transformers==4.40.1 torch numpy
```

### Rust Dependencies
```bash
cargo build --release
```

## Step 1: Capture Python Intermediates

Run the Python reference implementation to capture intermediate tensors:

```bash
python scripts/compare_intermediates.py \
    --model thuml/sundial-base-128m \
    --output /tmp/python_intermediates.npz \
    --seq-len 2880 \
    --batch-size 1
```

This will save all intermediate tensors to `/tmp/python_intermediates.npz`.

### Optional: Limit Layers
For faster debugging, you can stop after a specific number of layers:
```bash
python scripts/compare_intermediates.py --num-layers 2 --output /tmp/python_intermediates.npz
```

## Step 2: Capture Rust Intermediates

Run the Rust implementation with debug mode enabled:

```bash
# Enable debug output
export SUNDIAL_DEBUG=1

# Run with actual weights (when available)
cargo run --release --example forecast -- \
    --model-path <path-to-safetensors> \
    --seq-len 2880

# Or run the comparison example
cargo run --release --example compare_intermediates
```

This will save Rust tensors to `/tmp/<name>_rust.bin` files.

### Debug Environment Variables

- `SUNDIAL_DEBUG=1` - Enable debug output
- `SUNDIAL_DEBUG_LAYER=N` - Stop after layer N (for incremental debugging)

## Step 3: Compare Tensors

Compare the Python and Rust outputs:

```bash
python scripts/compare_tensors.py \
    --python /tmp/python_intermediates.npz \
    --rust /tmp/
```

This will show:
- Shape mismatches
- Maximum absolute difference
- Mean absolute difference
- Standard deviation of differences

## Interpreting Results

### Pass Criteria
- **Max diff < 1e-4**: ✓ PASS - Within numerical tolerance
- **Max diff < 1e-2**: ⚠ WARNING - Small differences, may accumulate
- **Max diff >= 1e-2**: ✗ FAIL - Significant deviation, needs investigation

### Example Output
```
============================================================
COMPARISON RESULTS
============================================================

patch_embed_output:
  Shape: [1, 180, 768]
  Max abs diff: 0.00001234
  Mean abs diff: 0.00000123
  ✓ PASS: Within tolerance (1e-4)

layer_0_output:
  Shape: [1, 180, 768]
  Max abs diff: 0.05678901
  Mean abs diff: 0.00234567
  ✗ FAIL: Large differences detected!
  Largest diff at index (0, 42, 156): 0.05678901
```

## Component Testing

For isolated component testing, use the compare_exact example:

```bash
cargo run --release --example compare_exact
```

This tests:
1. Patch Embedding
2. Rotary Positional Embeddings (RoPE)
3. Attention Layer
4. MLP Layer
5. Decoder Layer
6. Flow Sampling

All tests should pass with random weights. If they fail, there's a bug in the implementation.

## Common Issues and Fixes

### 1. RoPE Norm Not Preserved

**Symptom**: RoPE test fails, norm changes significantly

**Possible Causes**:
- Wrong dimension for rotation
- Incorrect cos/sin shape broadcasting
- Wrong order of operations

**Fix**:
```rust
// Ensure correct shape for broadcasting
let cos = cos.reshape((1, 1, seq_len, dim))?;
let sin = sin.reshape((1, 1, seq_len, dim))?;

// Apply rotation correctly
let q_embed = q.mul(&cos)?.add(&rotate_half(&q)?.mul(&sin)?)?;
```

### 2. Attention Weights Don't Sum to 1

**Symptom**: Attention output has unexpected values

**Possible Causes**:
- Incorrect softmax dimension
- Wrong causal mask broadcasting
- Missing scaling factor

**Fix**:
```rust
// Correct attention pattern
let scale = (query.dim(3)? as f64).sqrt().recip() as f32;
let attn_scores = query.matmul(&key.t()?)?.broadcast_mul(&Tensor::new(scale, query.device())?);
let attn_weights = candle_nn::ops::softmax(&attn_weights, 3)?;
```

### 3. Shape Mismatches

**Symptom**: Tensor shapes don't match expected values

**Possible Causes**:
- Wrong reshape dimensions
- Incorrect transpose order
- Missing broadcast operations

**Debug**:
```rust
println!("Shape: {:?}", tensor.dims());
```

### 4. NaN/Inf Values

**Symptom**: Output contains NaN or Inf

**Possible Causes**:
- Division by zero
- Log of negative number
- Overflow in exponentials

**Debug**:
```rust
let has_nan = tensor.flatten_all()?.to_vec1::<f32>()?
    .iter().any(|x| x.is_nan());
if has_nan {
    println!("NaN detected!");
}
```

## Debugging Strategy

### Top-Down Approach
1. Start with full model output comparison
2. Identify first layer with large deviation
3. Focus on that layer's components
4. Test individual components in isolation

### Bottom-Up Approach
1. Start with lowest-level components (RoPE, attention)
2. Verify each component works correctly
3. Build up to full layer
4. Test full model

### Recommended Workflow

1. **Run component tests**
   ```bash
   cargo run --release --example compare_exact
   ```

2. **Run full comparison**
   ```bash
   python scripts/compare_intermediates.py
   SUNDIAL_DEBUG=1 cargo run --release --example forecast
   python scripts/compare_tensors.py ...
   ```

3. **Identify failure point**
   - Look for first layer with max_diff > 1e-4

4. **Isolate the issue**
   - Use `SUNDIAL_DEBUG_LAYER=N` to stop before the problematic layer
   - Compare inputs to that layer

5. **Fix and verify**
   - Make targeted fix
   - Re-run tests
   - Verify no regressions

## Tensor Comparison Format

### Binary Format
Rust saves tensors in a custom binary format:
- 4 bytes: number of dimensions
- 4 bytes per dimension: dimension size
- 4 bytes per value: f32 little-endian

### Python Format
Python saves tensors in NumPy `.npz` format:
- Compressed archive with named arrays
- Each array is a numpy ndarray

## Performance Tips

1. **Limit layers** during debugging
   ```bash
   python scripts/compare_intermediates.py --num-layers 2
   ```

2. **Use smaller inputs** for faster iteration
   ```bash
   python scripts/compare_intermediates.py --seq-len 64
   ```

3. **Compare specific tensors** by name
   ```python
   # In compare_tensors.py, filter by name
   ```

## Troubleshooting

### Issue: No common tensors found

**Solution**: Check that both Python and Rust are saving tensors with the same names.

### Issue: Shape mismatch

**Solution**: Verify the tensor shapes match at each step. Use debug output to inspect shapes.

### Issue: Large differences everywhere

**Solution**: Check weight loading. Verify all tensors are loaded correctly from safetensors.

### Issue: Small differences everywhere

**Solution**: This may be due to numerical precision differences between frameworks. Check if max_diff < 1e-4.

## Additional Resources

- [Phase 1 Status Report](PHASE1_STATUS.md)
- [Original Plan](PLAN.md)
- [Sundial Python Implementation](https://github.com/thuml/Sundial)

---

*Last updated: 2026-04-03*
