# PRD: In-Memory Weight Loading for Sundial

**Date:** 2026-04-05  
**Status:** Draft  
**Author:** Sundial Team  
**Priority:** High  
**Estimated Effort:** Medium (2-3 days)

---

## 1. Overview

### 1.1 Problem Statement

The current Sundial Rust implementation extracts compressed model weights from embedded assets to disk before loading them into memory. This approach has several drawbacks:

1. **Unnecessary I/O Overhead**: Writing ~170MB compressed and reading ~490MB decompressed data to disk adds significant startup latency
2. **Disk Space Requirements**: Requires ~500MB of available disk space in temp directory
3. **Security Concerns**: Temporary files containing model weights exist on disk, even with restrictive permissions
4. **Cross-Platform Complexity**: Different temp directory handling across platforms
5. **Cleanup Reliability**: Risk of leftover temp files if cleanup fails

### 1.2 Goal

Eliminate disk extraction entirely by loading decompressed model weights directly into memory and passing them to the model loader. This will:

- Reduce startup time by eliminating disk I/O
- Remove disk space requirements for weight extraction
- Improve security by never writing weights to disk
- Simplify cross-platform code by removing temp directory management
- Maintain all existing functionality (external weights, hash verification)

### 1.3 Success Metrics

- **Startup Time**: Reduce model loading time by 30-50% (eliminate ~500MB disk write + read)
- **Memory Usage**: Increase peak memory by ~500MB during loading (acceptable trade-off)
- **Disk Space**: Remove ~500MB disk space requirement for temp extraction
- **Code Complexity**: Reduce weight loader code by 20-30% (remove extraction logic)
- **Security**: Eliminate temporary weight files from disk entirely

---

## 2. Current Implementation Analysis

### 2.1 Current Flow

```
build.rs
  ├── Compresses model.safetensors → model.safetensors.gz
  ├── Computes SHA256 hash of original
  └── Embeds via include_bytes!() macro

src/assets/mod.rs
  ├── WEIGHTS_COMPRESSED: &[u8] (gzip compressed, ~170MB)
  ├── CONFIG_JSON: &[u8] (config.json, ~1KB)
  └── MODEL_SHA256: &str (64 hex chars)

src/weights/loader.rs
  ├── WeightLoader::new()
  │   ├── Creates temp directory
  │   ├── Decompresses WEIGHTS_COMPRESSED → disk file
  │   ├── Writes config.json → disk file
  │   └── Verifies SHA256 hash of disk file
  │
  └── extract_with_progress()
      ├── Opens output file on disk
      ├── Decompresses gzip data
      └── Writes ~490MB to disk

src/model/loader.rs
  ├── load_safetensors(path)
  │   └── Reads weights from disk file
  └── Creates VarBuilder from disk-loaded tensors
```

### 2.2 Key Issues

1. **Double I/O**: Decompress to disk (~500MB write), then load from disk (~500MB read)
2. **Temp Directory Management**: Complex logic for custom temp paths, permissions, cleanup
3. **Race Conditions**: Temp files could be accessed by other processes before cleanup
4. **Error Handling**: Multiple failure points during extraction (disk full, permissions, etc.)

### 2.3 Existing Partial Solution

The code already has `WeightLoader::new_with_memory_weights()` that:
- Decompresses weights into memory
- Verifies hash from memory
- Stores in `model_weights: Option<Vec<u8>>`

**However**, this method is not used by the model loading pipeline. The model loader still expects file paths.

---

## 3. Proposed Solution

### 3.1 High-Level Design

```
build.rs (unchanged)
  └── Embeds compressed weights and config

src/assets/mod.rs (unchanged)
  └── Provides WEIGHTS_COMPRESSED, CONFIG_JSON, MODEL_SHA256

src/weights/loader.rs (modified)
  ├── WeightLoader::new()
  │   └── Check for external weights (unchanged)
  │
  ├── WeightLoader::new_with_memory_weights() (enhanced)
  │   ├── Decompress WEIGHTS_COMPRESSED → Vec<u8> in memory
  │   ├── Verify SHA256 hash from memory
  │   └── Parse CONFIG_JSON → SundialConfig
  │
  └── WeightLoader::load_into_candle() (NEW)
      ├── Takes decompressed weights from memory
      ├── Calls load_safetensors_from_bytes()
      └── Returns VarBuilder directly

src/model/loader.rs (modified)
  ├── load_safetensors_from_bytes(data, device) (enhanced)
  │   └── Parse safetensors from memory buffer
  │
  └── load_sundial_from_memory(weights_bytes, config, device) (NEW)
      ├── Decompress weights if needed
      ├── Verify integrity
      ├── Parse safetensors format
      └── Create VarBuilder with TensorMapBackend
```

### 3.2 Key Changes

#### 3.2.1 Remove Disk Extraction from Default Flow

**Before:**
```rust
let loader = WeightLoader::new()?;
let model_path = loader.model_path(); // Path to temp file
let tensors = load_safetensors(&model_path, &device)?;
```

**After:**
```rust
let loader = WeightLoader::new()?;
let tensors = loader.load_into_candle(&device)?; // Loads from memory
```

#### 3.2.2 Enhanced Memory Loading

```rust
impl WeightLoader {
    /// Load weights directly into Candle VarBuilder from memory
    pub fn load_into_candle(&self, device: &Device) -> Result<VarBuilder> {
        if let Some(weights) = &self.model_weights {
            // Already in memory
            load_safetensors_from_bytes(weights, device)
        } else {
            // Need to decompress first
            let mut decoder = GzDecoder::new(WEIGHTS_COMPRESSED);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;
            load_safetensors_from_bytes(&decompressed, device)
        }
    }
}
```

#### 3.2.3 Model Loader Integration

```rust
impl SundialModel {
    /// Load model from memory-resident weights
    pub fn load_from_memory(
        weights: &[u8],
        config: &SundialConfig,
        device: &Device,
    ) -> Result<Self> {
        let tensors = load_safetensors_from_bytes(weights, device)?;
        let vb = create_varbuilder(tensors, device)?;
        Self::new(config, vb)
    }
}
```

---

## 4. User Stories

### US-001: Memory-First Weight Loading

**As a** developer  
**I want** the default weight loading to use memory instead of disk  
**So that** startup is faster and no temp files are created

**Acceptance Criteria:**
- [ ] `WeightLoader::new()` loads weights into memory by default
- [ ] No temp directory is created for embedded weights
- [ ] Weights are decompressed and held in memory
- [ ] Model loading uses in-memory weights directly
- [ ] Hash verification works on in-memory data

### US-002: Backward Compatibility

**As a** user  
**I want** to still be able to use external weights via environment variables  
**So that** I can test with different model versions without recompiling

**Acceptance Criteria:**
- [ ] `SUNDIAL_MODEL_PATH` and `SUNDIAL_CONFIG_PATH` still work
- [ ] External weights are loaded from specified paths (disk-based)
- [ ] Embedded weights are used when no external paths are set
- [ ] Clear documentation on how to switch between modes

### US-003: Memory-Efficient Loading

**As a** user with limited memory  
**I want** an option to use disk extraction  
**So that** I can trade startup time for lower peak memory usage

**Acceptance Criteria:**
- [ ] `WeightLoader::new_with_disk_extraction()` method available
- [ ] Can be enabled via `SUNDIAL_USE_DISK=true` environment variable
- [ ] Default is memory loading, disk is opt-in
- [ ] Documentation explains memory vs disk tradeoffs

### US-004: Hash Verification

**As a** security-conscious user  
**I want** weights to be verified for integrity  
**So that** I know the model hasn't been tampered with

**Acceptance Criteria:**
- [ ] SHA256 hash verification works on in-memory data
- [ ] Verification happens before model loading
- [ ] Clear error message if hash doesn't match
- [ ] Test coverage for hash verification

---

## 5. Technical Specifications

### 5.1 API Changes

#### New Public API

```rust
// weights/loader.rs
impl WeightLoader {
    /// Load weights into memory and return a VarBuilder
    pub fn load_into_candle(&self, device: &Device) -> Result<VarBuilder>;
    
    /// Check if weights are loaded in memory
    pub fn has_memory_weights(&self) -> bool;
}

// model/loader.rs
pub fn load_sundial_from_memory(
    weights: &[u8],
    config: &SundialConfig,
    device: &Device,
) -> Result<SundialModel>;
```

#### Modified API

```rust
// weights/loader.rs - Remove these methods or mark deprecated
impl WeightLoader {
    // Remove or deprecate:
    // - model_path() - returns "<memory>" or actual path
    // - config_path() - returns "<memory>" or actual path
}
```

### 5.2 Memory Layout

```
Process Memory
├── Embedded Compressed Weights (const, ~170MB)
│   └── WEIGHTS_COMPRESSED
├── Decompressed Weights (Vec<u8>, ~490MB)
│   └── model_weights field
├── Candle Tensors (allocated on device, ~490MB)
│   └── Loaded from safetensors bytes
└── Model Structure (SundialModel, ~50MB)
    └── Tensor references
```

**Peak Memory**: ~1.2GB (compressed + decompressed + tensors + model)  
**Steady State**: ~540MB (tensors + model, decompressed can be dropped)

### 5.3 Performance Characteristics

| Operation | Current (Disk) | Proposed (Memory) | Improvement |
|-----------|----------------|-------------------|-------------|
| Decompression | 170MB read + 500MB write | 170MB read only | ~2x faster |
| Model Loading | 500MB read from disk | 0 disk I/O | ~3x faster |
| Total Startup | ~2-3 seconds | ~1-1.5 seconds | ~40% faster |
| Disk Space | 500MB required | 0 required | 100% reduction |

### 5.4 Error Handling

```rust
pub enum WeightLoadError {
    DecompressionFailed(String),
    HashVerificationFailed { expected: String, actual: String },
    SafetensorsParseError(String),
    TensorLoadError(String),
    InsufficientMemory(String),
}
```

---

## 6. Implementation Plan

### Phase 1: Core Memory Loading (Day 1)

**Tasks:**
1. [ ] Enhance `WeightLoader::new_with_memory_weights()` to be the default
2. [ ] Implement `load_into_candle()` method on `WeightLoader`
3. [ ] Update `load_safetensors_from_bytes()` to handle tensor name mapping
4. [ ] Add `load_sundial_from_memory()` function
5. [ ] Write unit tests for memory loading path

**Deliverables:**
- Working memory-based weight loading
- Unit tests passing
- No changes to public API yet

### Phase 2: Integration & Cleanup (Day 2)

**Tasks:**
1. [ ] Update model loading to use `load_into_candle()` by default
2. [ ] Remove or deprecate disk extraction code
3. [ ] Update `model_path()` and `config_path()` to return "<memory>"
4. [ ] Add environment variable `SUNDIAL_USE_DISK` for backward compatibility
5. [ ] Update integration tests

**Deliverables:**
- Default loading uses memory
- External weights still work
- Integration tests passing

### Phase 3: Optimization & Documentation (Day 3)

**Tasks:**
1. [ ] Profile memory usage and optimize (drop decompressed after loading)
2. [ ] Add benchmark comparing memory vs disk loading
3. [ ] Update documentation and examples
4. [ ] Add migration guide for users
5. [ ] Performance testing and validation

**Deliverables:**
- Performance benchmarks
- Updated documentation
- Release notes

---

## 7. Testing Strategy

### 7.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_memory_loading_succeeds() {
        let loader = WeightLoader::new_with_memory_weights().unwrap();
        assert!(loader.has_memory_weights());
    }

    #[test]
    fn test_hash_verification_memory() {
        let loader = WeightLoader::new_with_memory_weights().unwrap();
        // Should not panic - hash verified
    }

    #[test]
    fn test_load_into_candle() {
        let loader = WeightLoader::new_with_memory_weights().unwrap();
        let device = Device::Cpu;
        let vb = loader.load_into_candle(&device).unwrap();
        assert!(vb.is_ok());
    }

    #[test]
    fn test_sundial_from_memory() {
        let weights = decompress_weights(); // Helper function
        let config = load_config().unwrap();
        let device = Device::Cpu;
        let model = load_sundial_from_memory(&weights, &config, &device).unwrap();
        assert!(model.is_ok());
    }
}
```

### 7.2 Integration Tests

```rust
#[test]
fn test_full_inference_memory() {
    // Load model from memory
    // Run inference
    // Verify output matches expected values
}

#[test]
fn test_external_weights_still_work() {
    // Set SUNDIAL_MODEL_PATH and SUNDIAL_CONFIG_PATH
    // Verify model loads from specified paths
}

#[test]
fn test_disk_fallback() {
    // Set SUNDIAL_USE_DISK=true
    // Verify weights are extracted to disk
}
```

### 7.3 Performance Benchmarks

```rust
#[cfg(feature = "bench")]
mod benchmarks {
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_memory_loading(c: &mut Criterion) {
        c.bench_function("load_weights_memory", |b| {
            b.iter(|| {
                let loader = WeightLoader::new_with_memory_weights().unwrap();
                let device = Device::Cpu;
                black_box(loader.load_into_candle(&device).unwrap());
            })
        });
    }

    fn bench_disk_loading(c: &mut Criterion) {
        c.bench_function("load_weights_disk", |b| {
            b.iter(|| {
                std::env::set_var("SUNDIAL_USE_DISK", "true");
                let loader = WeightLoader::new().unwrap();
                let device = Device::Cpu;
                black_box(loader.load_into_candle(&device).unwrap());
                std::env::remove_var("SUNDIAL_USE_DISK");
            })
        });
    }

    criterion_group!(benches, bench_memory_loading, bench_disk_loading);
    criterion_main!(benches);
}
```

---

## 8. Risks and Mitigations

### 8.1 Memory Usage

**Risk:** Peak memory increases by ~500MB during loading  
**Mitigation:**
- Drop decompressed weights after loading tensors
- Document memory requirements clearly
- Provide `SUNDIAL_USE_DISK` fallback for low-memory systems

### 8.2 Breaking Changes

**Risk:** Code relying on `model_path()` returning a valid path breaks  
**Mitigation:**
- Return `"<memory>"` string for in-memory mode
- Keep `model_path()` but document it may return virtual path
- Provide migration guide

### 8.3 Safetensors Parsing

**Risk:** Custom parsing may have bugs compared to candle's file-based loader  
**Mitigation:**
- Use official `safetensors` crate for parsing
- Extensive testing with known-good weights
- Compare tensor values with disk-based loading

---

## 9. Dependencies

### 9.1 Required Crates

```toml
[dependencies]
# Already present
flate2 = "1.0"          # For gzip decompression
sha2 = "0.10"           # For hash verification
safetensors = "0.4"     # For parsing safetensors format
candle-core = "0.8"     # For tensor operations
candle-nn = "0.8"       # For VarBuilder
anyhow = "1.0"          # For error handling
tracing = "0.1"         # For logging

# Optional for disk fallback
tempfile = "3.10"       # Only used if SUNDIAL_USE_DISK=true
```

### 9.2 No New Dependencies

This change uses existing dependencies; no new crates needed.

---

## 10. Documentation Updates

### 10.1 README.md

Add section:
```markdown
## Model Loading

Sundial loads model weights directly into memory for fast startup:

```rust
use sundial_rust::{WeightLoader, SundialModel};

// Load from embedded weights (memory)
let loader = WeightLoader::new()?;
let model = SundialModel::load(&loader, &device)?;
```

### Environment Variables

- `SUNDIAL_USE_DISK=true` - Use disk extraction instead of memory
- `SUNDIAL_MODEL_PATH` - Path to external model.safetensors
- `SUNDIAL_CONFIG_PATH` - Path to external config.json
```

### 10.2 API Documentation

Update doc comments for:
- `WeightLoader::new()` - Clarify memory-first approach
- `WeightLoader::model_path()` - Document "<memory>" return value
- `WeightLoader::new_with_memory_weights()` - Mark as internal

### 10.3 Migration Guide

Create `MIGRATION.md`:
```markdown
# Migration to Memory-First Loading

## What Changed

Weight loading now uses memory instead of disk by default.

## Breaking Changes

- `WeightLoader::model_path()` may return `"<memory>"` instead of a real path
- If you need file paths, use `SUNDIAL_USE_DISK=true`

## Migration Steps

1. Update code that assumes `model_path()` returns a real file
2. Use `WeightLoader::load_into_candle()` directly
3. Or set `SUNDIAL_USE_DISK=true` for backward compatibility
```

---

## 11. Success Criteria

### 11.1 Functional Requirements

- [ ] Model loads successfully from memory
- [ ] Hash verification works correctly
- [ ] External weights via env vars still work
- [ ] All existing tests pass
- [ ] No temp files created during loading

### 11.2 Performance Requirements

- [ ] Startup time reduced by ≥30%
- [ ] Peak memory ≤1.2GB
- [ ] No disk I/O during weight loading (when using embedded weights)

### 11.3 Quality Requirements

- [ ] 100% unit test coverage on new code
- [ ] Integration tests pass
- [ ] Benchmarks show expected improvement
- [ ] Documentation complete and accurate

---

## 12. Future Enhancements

### 12.1 Memory Mapping

Consider memory-mapped loading for very large models:
- Use `memmap2` crate
- Load weights without full decompression into RAM
- Trade-off: slower access, lower memory usage

### 12.2 Lazy Loading

Load only required layers on-demand:
- Parse safetensors index
- Load tensors as needed
- Good for models with selective layer usage

### 12.3 Compression Optimization

Explore better compression:
- Zstandard instead of gzip
- Quantized weights (INT8, FP16)
- Decompress on-the-fly during loading

---

## Appendix A: File Changes Summary

| File | Changes |
|------|---------|
| `src/weights/loader.rs` | Major refactor - make memory loading default |
| `src/model/loader.rs` | Add `load_safetensors_from_bytes()` enhancement |
| `src/assets/mod.rs` | No changes |
| `build.rs` | No changes |
| `Cargo.toml` | No changes (reuse existing deps) |
| `tests/` | Add memory loading tests |
| `docs/` | Update documentation |

---

## Appendix B: Reference Implementation

See `src/weights/loader.rs` for existing `new_with_memory_weights()` implementation - this will be expanded and made default.
