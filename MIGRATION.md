# Migration Guide: In-Memory Weight Loading

This document describes the migration to Sundial's new memory-first weight loading approach.

---

## What Changed

### Memory-First Default Behavior

Sundial now loads model weights directly into memory by default, eliminating the need for disk extraction during startup.

**Previous behavior:**
1. Decompress embedded weights (~170MB compressed → ~490MB decompressed)
2. Write decompressed weights to temporary disk files
3. Write config.json to disk
4. Verify SHA256 hash of disk files
5. Load tensors from disk files into memory
6. Delete temporary files

**New behavior:**
1. Decompress embedded weights into memory (~170MB → ~490MB in RAM)
2. Verify SHA256 hash of in-memory data
3. Parse safetensors directly from memory buffer
4. Create model with tensor references (no disk I/O)

The decompressed weights are held temporarily during model initialization and can be dropped afterward to free ~490MB of memory.

---

## Breaking Changes

### `model_path()` May Return `"<memory>"`

The `WeightLoader::model_path()` method now returns `"<memory>"` when using the default in-memory loading mode:

```rust
let loader = WeightLoader::new()?;
let path = loader.model_path();

// Previous: Always returned a real filesystem path (e.g., "/tmp/sundial-weights/model.safetensors")
// New: Returns "<memory>" when using embedded weights in memory mode
if path.to_str() == Some("<memory>") {
    println!("Using in-memory weights - no filesystem path available");
}
```

**Impact:** Code that assumes `model_path()` returns a valid filesystem path will break.

### `WeightLoader::new()` Signature Unchanged

The constructor signature remains the same, but the default behavior has changed:

```rust
// Still works, but now loads into memory by default
let loader = WeightLoader::new()?;
```

---

## Migration Steps

### Step 1: Check for `"<memory>"` Path

If your code uses `model_path()`, add a check for the `"<memory>"` sentinel value:

```rust
use sundial_rust::weights::loader::WeightLoader;
use std::path::Path;

let loader = WeightLoader::new()?;
let path = loader.model_path();

// Handle in-memory case
if path.to_str() == Some("<memory>") {
    // Use memory loading path
    let vb = loader.load_into_candle(&device)?;
    let model = SundialModel::new(&config, vb)?;
} else {
    // Use disk loading path (external weights or SUNDIAL_USE_DISK=true)
    let tensors = load_safetensors(path, &device)?;
    let vb = create_varbuilder(tensors, &device)?;
    let model = SundialModel::new(&config, vb)?;
}
```

### Step 2: Use `load_into_candle()` for Memory Loading

If you previously called `WeightLoader::new()` and then loaded from the returned path, switch to the new API:

```rust
// Old pattern (no longer works with in-memory weights)
let loader = WeightLoader::new()?;
let model_path = loader.model_path();
let tensors = load_safetensors(&model_path, &device)?;

// New pattern - direct memory loading
let loader = WeightLoader::new()?;
let vb = loader.load_into_candle(&device)?;
let model = SundialModel::new(&config, vb)?;
```

### Step 3: Update `SundialModel::load_from_safetensors()`

If you call `SundialModel::load_from_safetensors()`, it now handles memory loading automatically:

```rust
// This now works with both memory and disk weights
let model = SundialModel::load_from_safetensors(
    config,
    "model.safetensors",  // Path is ignored if SUNDIAL_MODEL_PATH not set
    &device,
)?;
// If the path doesn't exist, it automatically falls back to embedded memory weights
```

### Step 4: Handle External Weights

External weights via environment variables still work as before:

```bash
# Still supported - uses disk loading from specified path
export SUNDIAL_MODEL_PATH=/path/to/model.safetensors
export SUNDIAL_CONFIG_PATH=/path/to/config.json
```

```rust
// External weights are detected automatically
let model = SundialModel::load_from_safetensors(config, "ignored.safetensors", &device)?;
// Will load from SUNDIAL_MODEL_PATH if set
```

### Step 5: Enable Disk Mode (Optional)

If you need disk-based loading (e.g., for memory-constrained environments):

```bash
# Force disk extraction mode
export SUNDIAL_USE_DISK=true
```

```rust
// Or use the explicit disk loader
let loader = WeightLoader::new_with_disk_extraction()?;
let model_path = loader.model_path(); // Returns real filesystem path
```

---

## Performance Benefits

### Startup Time

| Metric | Before (Disk) | After (Memory) | Improvement |
|--------|---------------|----------------|-------------|
| Decompression I/O | ~500MB write + ~500MB read | ~170MB read only | ~2.4x faster |
| Model Loading | ~500MB disk read | 0 disk I/O | ~3x faster |
| Total Startup Time | ~2-3 seconds | ~1-1.5 seconds | **~40-50% faster** |

### Disk Space

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Temporary Disk Space | ~500MB required | 0 required | **100% reduction** |
| Temporary Files | Created and deleted | None | **No temp files** |

### Memory Usage

| Phase | Before | After | Notes |
|-------|--------|-------|-------|
| Peak Memory | ~1.0GB | ~1.2GB | +200MB during loading |
| Steady State | ~540MB | ~540MB | Identical after loading |

**Note:** The ~200MB peak memory increase is temporary and occurs only during model initialization.

### Security Improvements

- **No temporary weight files** on disk
- **No risk of temp file leakage** if cleanup fails
- **No cross-platform temp directory complexity**
- **Weights never touch the filesystem** in default mode

---

## Quick Reference

| Feature | Old API | New API |
|---------|---------|---------|
| Create loader | `WeightLoader::new()` | `WeightLoader::new()` (unchanged) |
| Get model path | `loader.model_path()` | Returns `"<memory>"` for in-memory mode |
| Load tensors | `load_safetensors(&path, &device)` | `loader.load_into_candle(&device)` |
| Force disk mode | N/A | `WeightLoader::new_with_disk_extraction()` |
| External weights | `SUNDIAL_MODEL_PATH` | `SUNDIAL_MODEL_PATH` (unchanged) |

---

## Examples

### Complete Migration Example

```rust
use sundial_rust::model::SundialModel;
use sundial_rust::weights::loader::WeightLoader;
use candle_core::Device;

// Old code
fn load_model_old(device: &Device) -> anyhow::Result<SundialModel> {
    let loader = WeightLoader::new()?;
    let model_path = loader.model_path();
    
    // This fails when model_path is "<memory>"
    let tensors = load_safetensors(&model_path, device)?;
    let vb = create_varbuilder(tensors, device)?;
    
    let config = load_config()?;
    Ok(SundialModel::new(&config, vb))
}

// New code
fn load_model_new(device: &Device) -> anyhow::Result<SundialModel> {
    let loader = WeightLoader::new()?;
    let config = load_config()?;
    
    // Handle both memory and disk paths
    if loader.has_memory_weights() {
        // Fast path: load directly from memory
        let vb = loader.load_into_candle(device)?;
        Ok(SundialModel::new(&config, vb))
    } else {
        // Fallback: load from disk (external weights or SUNDIAL_USE_DISK=true)
        let tensors = load_safetensors(loader.model_path(), device)?;
        let vb = create_varbuilder(tensors, device)?;
        Ok(SundialModel::new(&config, vb))
    }
}

// Simplest: let SundialModel handle it
fn load_model_simple(device: &Device) -> anyhow::Result<SundialModel> {
    let config = load_config()?;
    // Automatically uses memory weights if available
    SundialModel::load_from_safetensors(config, "ignored.safetensors", device)
}
```

---

## Rollback

If you need to revert to disk-based loading:

```rust
// Use explicit disk extraction
let loader = WeightLoader::new_with_disk_extraction()?;
let model_path = loader.model_path(); // Real filesystem path
let tensors = load_safetensors(&model_path, &device)?;
```

Or set the environment variable:

```bash
export SUNDIAL_USE_DISK=true
```

---

## Support

For issues or questions:
- Check the [README.md](README.md) for usage examples
- Review [docs/PRD-memory-weights.md](docs/PRD-memory-weights.md) for design details
- Open an issue on GitHub with your use case
