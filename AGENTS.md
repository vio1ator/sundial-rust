# Sundial Rust - Agent Guidelines

Time series forecasting project using the Sundial model implemented in Rust with Candle ML framework.

## Quick Start Commands

After ANY code change, run quality checks:

```bash
# Run all tests
cargo test

# Lint with clippy
cargo clippy

# Build the project
cargo build

# Check formatting
cargo fmt --check
```

## Recommended Workflow

1. Make changes
2. Run `cargo clippy` to catch issues
3. Run `cargo test` to verify functionality
4. Run `cargo fmt` to format code
5. Commit changes

## Project Structure

```
sundial-rust/
├── src/
│   ├── assets/          # Embedded weights and configuration (new in US-002)
│   ├── data/            # Data loading and preprocessing
│   ├── flow/            # Training/inference pipeline
│   ├── model/           # Sundial model implementation
│   └── lib.rs           # Public API and debug utilities
├── scripts/             # Python comparison scripts
├── archive-docs/        # Historical implementation docs
├── weights/             # Model weights (config.json + model.safetensors)
└── build.rs             # Build script for weight compression
```

## Key Conventions

### Error Handling
- Use `anyhow::Result` for application-level errors
- Use `thiserror` for library-specific error types
- Always return `Result` types from fallible operations

### Logging
- Use `tracing` crate for structured logging
- Use appropriate span levels: `info!`, `debug!`, `warn!`, `error!`
- Include contextual information in log messages

### Asset Module (US-002)
- Create `src/assets/mod.rs` module for embedded weights
- Use `include_bytes!(env!("WEIGHTS_PATH"))` to embed compressed weights
- Use `include_bytes!("../../weights/config.json")` for config
- Use `env!("MODEL_SHA256")` for integrity hash constant
- Export public constants and helper functions from lib.rs
- Write unit tests to verify embedded assets are non-empty and valid

### Build Script Patterns

#### Compressing Weights
- Build scripts use `OUT_DIR` environment variable (not `CARGO_OUT_DIR`)
- Compress model weights during build using `flate2::GzEncoder`
- Compute SHA256 hash for integrity verification using `sha2` crate
- Set `WEIGHTS_PATH` and `MODEL_SHA256` environment variables for the main build
- Place compressed weights in a subdirectory under `OUT_DIR` for organization
- Use `cargo:rerun-if-changed` to rebuild when weights change

#### Embedding Assets
- Use `env!("ENV_VAR")` macro to reference build script environment variables
- Use `include_bytes!` macro for compile-time binary data embedding
- Define constants as `pub const NAME: &str = env!("VAR")` for strings
- Define constants as `pub const NAME: &[u8] = include_bytes!(...)` for binary data

## Key Conventions

### Error Handling
- Use `anyhow::Result` for application-level errors
- Use `thiserror` for library-specific error types
- Always return `Result` types from fallible operations

### Logging
- Use `tracing` crate for structured logging
- Use appropriate span levels: `info!`, `debug!`, `warn!`, `error!`
- Include contextual information in log messages

### Tensor Operations
- Use `candle-core`, `candle-nn`, `candle-transformers` for all ML operations
- Use `debug_tensor()` from `debug_utils` for tensor inspection during debugging
- Compare with Python implementation using `compare_tensors()` when debugging

### Data Handling
- Use `polars` for data loading and manipulation
- Use `candle` tensors for model operations
- Convert between polars DataFrames and candle tensors at boundaries

### Code Style
- Follow Rust 2021 edition conventions
- Use named imports for clarity
- Keep functions focused and single-purpose
- Document public APIs with rustdoc comments

## Testing Patterns

- Unit tests in `#[cfg(test)]` modules within source files
- Integration tests in `tests/` directory (if created)
- Use `criterion` for benchmarking performance-critical code
- Test tensor shapes and values explicitly

## Debugging

When debugging model discrepancies:
1. Use `debug_tensor()` to inspect tensor statistics
2. Use `compare_tensors()` to compare Rust vs Python outputs
3. Save intermediate tensors to `/tmp/` for offline analysis
4. Check weight loading matches Python implementation

## Dependencies

- **ML**: candle-core, candle-nn, candle-transformers (v0.8)
- **Data**: polars (v0.45), ndarray (v0.15)
- **Async**: tokio (v1)
- **CLI**: clap (v4.5)
- **Error handling**: anyhow (v1.0), thiserror (v2.0)
- **Logging**: tracing, tracing-subscriber
