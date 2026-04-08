# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

### Tensor Comparison Pattern
Use `ComparisonStats` struct to compute max_diff, mean_diff, and std_diff for tensor comparisons:
```rust
let stats = ComparisonStats::new(&actual, &expected);
// stats.max_diff, stats.mean_diff, stats.std_diff, stats.shape
```

### Layer-by-Layer Debugging Pattern
Use `SUNDIAL_DEBUG=1` environment variable to save intermediate tensors to `/tmp/`:
```rust
// In transformer.rs::forward()
if std::env::var("SUNDIAL_DEBUG").is_ok() {
    debug_utils::save_tensor_to_bin("patch_embed_output", &hidden_states)?;
}
```

### NPY File Loading Pattern
Parse numpy .npy files manually to load Python reference data:
```rust
fn load_npy_as_tensor(path: &str) -> Result<Tensor, String> {
    // Parse header for shape and dtype
    // Extract data and reshape
}
```

---

## 2026-04-06 - US-001
- **What was implemented:**
  - Created systematic comparison framework in `tests/layer_comparison.rs`
  - Added `ComparisonStats` struct to compute max_diff, mean_diff, std_diff for each component
  - Implemented comparison tests for:
    - `test_patch_embed_comparison()`: Compares patch embedding output
    - `test_rope_comparison()`: Compares RoPE outputs for each of 12 layers (Q and K after RoPE)
    - `test_block_comparison()`: Compares each decoder block output
    - `test_flow_matching_comparison()`: Compares flow matching samples
  - Added `test_rope_detailed_comparison()`: Standalone RoPE test with detailed statistics
  - Added `test_layer_by_layer_comparison()`: Comprehensive layer-by-layer comparison test
  - Added `test_end_to_end_with_intermediates()`: End-to-end test with intermediate saving
  - Added unit tests for `ComparisonStats` and `compare_tensors` functions
  - Implemented JSON report generation for comparison results
  - Fixed `broadcast_sub` tensor operation to use scalar tensor wrapper

- **Files changed:**
  - `tests/layer_comparison.rs`: Created comprehensive comparison framework
  - `.ralph-tui/progress.md`: Updated with implementation details and patterns

- **Learnings:**
  - Candle's `broadcast_sub` requires a tensor argument, not a scalar - need to wrap scalar in `Tensor::new()`
  - NPY file format parsing requires careful header parsing to extract shape and data type
  - The framework successfully identifies discrepancies: RoPE test shows max_diff=7.66, confirming the issues noted in PLAN.md
  - All comparison tests are marked with `#[ignore]` attribute to run only when Python reference data is available
  - Unit tests for the comparison framework pass successfully, verifying the statistics computation logic
  - `cargo clippy` passes with no errors (only warnings for unused code)

---

