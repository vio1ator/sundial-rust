# PRD: Fix End-to-End Test Discrepancies Between Rust and Python Sundial Implementations

## Overview

This PRD addresses the critical issue of discrepancies between Rust and Python implementations of the Sundial model, as identified in PLAN.md. The goal is to systematically investigate and fix the root causes causing the end-to-end test to fail, ensuring parity between the two implementations across all model components.

## Goals

- Identify and fix the root cause of discrepancies between Rust and Python Sundial implementations
- Achieve test success criteria: max_diff < 1.0 and mean_diff < 0.5
- Systematically investigate all model components (patch embed, RoPE, layer blocks, flow matching)
- Establish debugging infrastructure for future discrepancy investigations
- Ensure all quality checks pass after fixes

## Quality Gates

These commands must pass for every user story:
- `cargo test` - Run all tests
- `cargo clippy` - Linting with clippy

## User Stories

### US-001: Set up systematic comparison framework
**Description:** As a developer, I want a structured comparison framework that outputs detailed statistics for each model component so that I can identify where discrepancies occur.

**Acceptance Criteria:**
- [ ] Create test that compares Rust vs Python outputs at each model layer
- [ ] Output max_diff, mean_diff, and std_diff for each component
- [ ] Generate comparison reports for: patch_embed, each RoPE layer, each block, flow matching
- [ ] Test currently fails with discrepancies as shown in PLAN.md

### US-002: Investigate patch embed layer
**Description:** As a developer, I want to verify the patch embedding layer produces identical outputs so that I can rule it out as a source of discrepancy.

**Acceptance Criteria:**
- [ ] Compare patch_embed output between Rust and Python
- [ ] Verify weight loading matches exactly (same values, same shapes)
- [ ] Verify projection operation produces identical results
- [ ] Document findings with specific metrics

### US-003: Investigate RoPE implementation
**Description:** As a developer, I want to verify the Rotary Positional Embedding implementation so that I can confirm or rule out it as the primary source of discrepancy.

**Acceptance Criteria:**
- [ ] Compare RoPE calculations layer by layer
- [ ] Verify frequency calculations match Python implementation
- [ ] Verify rotation matrix computation is identical
- [ ] Check for any differences in tensor ordering or broadcasting
- [ ] Document specific discrepancies found

### US-004: Investigate layer blocks
**Description:** As a developer, I want to verify each transformer block produces matching outputs so that I can identify which specific blocks diverge.

**Acceptance Criteria:**
- [ ] Compare each block's attention mechanism output
- [ ] Compare each block's feed-forward network output
- [ ] Verify layer normalization is applied identically
- [ ] Check for residual connection implementation differences
- [ ] Document per-block discrepancy metrics

### US-005: Investigate flow matching head
**Description:** As a developer, I want to verify the flow matching head produces correct outputs so that I can ensure the final layer is not causing discrepancies.

**Acceptance Criteria:**
- [ ] Compare flow matching head output between implementations
- [ ] Verify final projection weights match
- [ ] Check for any activation function differences
- [ ] Document final output discrepancy metrics

### US-006: Fix identified discrepancies
**Description:** As a developer, I want to fix all identified discrepancies so that the end-to-end test passes with acceptable tolerance levels.

**Acceptance Criteria:**
- [ ] Apply fixes based on investigation findings
- [ ] Verify max_diff < 1.0 and mean_diff < 0.5
- [ ] Ensure all intermediate layers match within tolerance
- [ ] Run full test suite to confirm no regressions

### US-007: Add regression tests
**Description:** As a developer, I want to add regression tests that catch future discrepancies early so that parity is maintained.

**Acceptance Criteria:**
- [ ] Create automated tests that compare Rust vs Python outputs
- [ ] Set appropriate thresholds for max_diff and mean_diff
- [ ] Add CI integration to run comparison tests
- [ ] Document debugging procedures for future investigations

## Functional Requirements

- FR-1: Comparison framework must output detailed statistics for each model component
- FR-2: Weight loading must be verified to use identical values and shapes
- FR-3: RoPE implementation must match Python reference exactly
- FR-4: Each transformer block must produce matching outputs within tolerance
- FR-5: Final flow matching output must meet success criteria (max_diff < 1.0, mean_diff < 0.5)
- FR-6: All fixes must pass cargo test and cargo clippy

## Non-Goals (Out of Scope)

- Performance optimization of the Sundial model
- Adding new model features or capabilities
- Porting additional Python components not in PLAN.md
- Refactoring existing code beyond what's needed to fix discrepancies
- Changing test thresholds - maintain current requirements

## Technical Considerations

- Use existing tensor comparison utilities from the codebase
- Reference Python implementation in scripts/ for comparison
- Follow Candle ML patterns for tensor operations
- Ensure weight files are loaded identically in both implementations
- Pay special attention to:
  - Tensor dimension ordering (Candle vs PyTorch conventions)
  - RoPE frequency calculation formulas
  - Layer normalization epsilon values
  - Attention mask implementations

## Success Metrics

- End-to-end test passes with max_diff < 1.0 and mean_diff < 0.5
- All intermediate layer outputs match within acceptable tolerance
- All quality gates pass (cargo test, cargo clippy)
- Documentation of root causes and fixes applied
- Regression tests in place to prevent future discrepancies

## Open Questions

- What is the acceptable tolerance for floating point differences between Rust and Python?
- Are there any known differences in the mathematical libraries (e.g., different BLAS implementations)?
- Should we aim for exact bitwise equality or is numerical equivalence sufficient?
- Are there any specific RoPE variants or configurations being used that need clarification?