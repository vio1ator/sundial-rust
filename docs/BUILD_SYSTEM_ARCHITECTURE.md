# Sundial Rust - Build System Architecture

## Overview

This document defines the build system architecture for Sundial Rust, a time series forecasting model. The build system is designed to support cross-platform compilation across macOS, Linux, and Windows on both x86_64 and aarch64 architectures.

## Build System Components

### 1. Cargo.toml - Project Configuration

**Responsibilities:**
- Define project metadata (name, version, edition)
- Declare dependencies (runtime and build-time)
- Configure build profiles (dev, release)
- Define feature flags for conditional compilation
- Specify target-specific dependencies

**Key Sections:**

```toml
[package]
# Project metadata
name = "sundial-rust"
version = "0.1.0"
edition = "2021"

[dependencies]
# Runtime dependencies
candle-core = "0.8"          # ML framework
candle-nn = "0.8"            # Neural network API
candle-transformers = "0.8"  # Transformer models
polars = "0.45"              # Data manipulation
tokio = "1"                  # Async runtime

[build-dependencies]
# Build-time dependencies
flate2 = "1.0"               # Weight compression
sha2 = "0.10"                # Hash computation

[profile.release]
# Release build optimization
opt-level = 3
lto = true
```

### 2. build.rs - Build Script

**Responsibilities:**
- Compress model weights at build time
- Compute SHA256 hash for integrity verification
- Set environment variables for embedded assets
- Trigger rebuild when weights change

**Key Operations:**

```rust
// 1. Detect weights file location
let source_weights = Path::new("weights/model.safetensors");

// 2. Compute integrity hash
let sha256_hash = compute_sha256(&source_weights)?;

// 3. Compress weights for embedding
compress_weights(&source_weights, &compressed_path)?;

// 4. Set environment variables for compilation
println!("cargo:rustc-env=WEIGHTS_PATH={}", weights_path);
println!("cargo:rustc-env=MODEL_SHA256={}", sha256_hash);

// 5. Trigger rebuild on changes
println!("cargo:rerun-if-changed={}", source_weights.display());
```

### 3. Platform Detection and Configuration

**Responsibilities:**
- Detect target platform at compile time
- Apply platform-specific configurations
- Enable/disable platform-specific features

**Platform Detection Patterns:**

```rust
// Compile-time platform detection
#[cfg(target_os = "windows")]
// Windows-specific code

#[cfg(target_os = "macos")]
// macOS-specific code

#[cfg(target_os = "linux")]
// Linux-specific code

// Architecture detection
#[cfg(target_arch = "x86_64")]
// x86_64-specific code

#[cfg(target_arch = "aarch64")]
// ARM64-specific code

// Family detection
#[cfg(target_family = "unix")]
// Unix-family systems (macOS, Linux)

#[cfg(target_family = "windows")]
// Windows systems
```

## Build Output Directory Structure

### Native Build Output

```
target/
├── debug/                    # Debug builds
│   ├── sundial-rust         # Binary (Unix)
│   ├── sundial-rust.exe     # Binary (Windows)
│   └── build/
│       └── sundial-rust-<hash>/
│           └── out/
│               └── embedded_weights/
│                   ├── model.safetensors.gz
│                   └── model.safetensors.sha256
└── release/                  # Release builds
    ├── sundial-rust
    └── build/
        └── sundial-rust-<hash>/
            └── out/
                └── embedded_weights/
```

### Cross-Compilation Output

```
target/
├── x86_64-unknown-linux-gnu/
│   └── release/
│       └── sundial-rust      # Linux x86_64 binary
├── x86_64-unknown-linux-musl/
│   └── release/
│       └── sundial-rust      # Static Linux x86_64 binary
├── aarch64-unknown-linux-gnu/
│   └── release/
│       └── sundial-rust      # Linux ARM64 binary
├── x86_64-pc-windows-msvc/
│   └── release/
│       └── sundial-rust.exe  # Windows x86_64 binary
├── x86_64-apple-darwin/
│   └── release/
│       └── sundial-rust      # macOS Intel binary
└── aarch64-apple-darwin/
    └── release/
        └── sundial-rust      # macOS Apple Silicon binary
```

## Platform-Specific Configurations

### Linux (glibc)

**Target:** `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`

**Requirements:**
- GNU toolchain (gcc, g++)
- glibc libraries

**Configuration:**
```toml
# .cargo/config.toml
[target.x86_64-unknown-linux-gnu]
linker = "x86_64-linux-gnu-gcc"
```

### Linux (musl - Static)

**Target:** `x86_64-unknown-linux-musl`

**Requirements:**
- musl cross-compilation toolchain

**Configuration:**
```toml
[target.x86_64-unknown-linux-musl]
linker = "x86_64-linux-musl-gcc"
rustflags = ["-C", "link-arg=-static"]
```

### macOS

**Targets:** `x86_64-apple-darwin`, `aarch64-apple-darwin`

**Requirements:**
- Xcode Command Line Tools
- macOS SDK

**Notes:**
- Native builds use system toolchain
- No additional configuration required

### Windows

**Target:** `x86_64-pc-windows-msvc`

**Requirements:**
- Visual Studio Build Tools (MSVC)
- Windows SDK

**Notes:**
- Use MSVC toolchain for native builds
- GNU toolchain available via `x86_64-pc-windows-gnu`

## Environment Variables

### Build-Time Variables (set by build.rs)

| Variable | Description | Value Type |
|----------|-------------|------------|
| `WEIGHTS_PATH` | Path to compressed weights file | String path |
| `MODEL_SHA256` | SHA256 hash of original weights | Hex string |

### Cargo Environment Variables (auto-provided)

| Variable | Description |
|----------|-------------|
| `OUT_DIR` | Output directory for build artifacts |
| `CARGO_MANIFEST_DIR` | Project root directory |
| `TARGET` | Target triple (e.g., `x86_64-apple-darwin`) |
| `HOST` | Host platform triple |
| `PROFILE` | Build profile (`debug` or `release`) |

## Build Workflow

### 1. Build Script Execution

```
┌─────────────────────────────────────────────────────────────┐
│                    cargo build                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    build.rs                                 │
│  1. Compute SHA256 of weights/model.safetensors            │
│  2. Compress weights to OUT_DIR/embedded_weights/           │
│  3. Set WEIGHTS_PATH and MODEL_SHA256 env vars             │
│  4. Emit rerun-if-changed directives                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 rustc Compilation                           │
│  - Embed weights via include_bytes!()                       │
│  - Access SHA256 via env!("MODEL_SHA256")                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Final Binary                               │
│  - Weights embedded in executable                           │
│  - Integrity hash available at compile time                 │
└─────────────────────────────────────────────────────────────┘
```

### 2. Asset Loading at Runtime

```rust
// Embedded weights loaded at compile time
const COMPRESSED_WEIGHTS: &[u8] = include_bytes!(env!("WEIGHTS_PATH"));

// Integrity verification
const EXPECTED_SHA256: &str = env!("MODEL_SHA256");

// Runtime decompression
let decompressed = gzip_decode(COMPRESSED_WEIGHTS)?;
verify_sha256(&decompressed, EXPECTED_SHA256)?;
```

## Quality Gates

All builds must pass these checks:

```bash
# Build verification
cargo build

# Unit tests
cargo test

# Linting
cargo clippy

# Formatting
cargo fmt --check
```

## Cross-Compilation Commands

### Using `cross` (Recommended)

```bash
# Install cross
cargo install cross

# Linux (glibc)
cross build --target x86_64-unknown-linux-gnu --release

# Linux (musl, static)
cross build --target x86_64-unknown-linux-musl --release

# Windows
cross build --target x86_64-pc-windows-msvc --release
```

### Native Builds

```bash
# macOS (current platform)
cargo build --release

# Add target first
rustup target add <target-triple>

# Build for specific target
cargo build --target <target-triple> --release
```

## Asset Module Integration (US-002)

The build system integrates with the asset module for embedded weights:

```rust
// src/assets/mod.rs
pub const WEIGHTS_PATH: &str = env!("WEIGHTS_PATH");
pub const MODEL_SHA256: &str = env!("MODEL_SHA256");
pub const COMPRESSED_WEIGHTS: &[u8] = include_bytes!(WEIGHTS_PATH);

// src/lib.rs
pub mod assets;

pub fn get_model_weights() -> &'static [u8] {
    assets::COMPRESSED_WEIGHTS
}

pub fn get_weights_hash() -> &'static str {
    assets::MODEL_SHA256
}
```

## Build Script Error Handling

The `build.rs` script defines custom error types for clear diagnostics:

```rust
enum BuildError {
    MissingWeights(PathBuf),      // Weights file not found
    FileOpen(String, PathBuf),    // Cannot open file
    FileMetadata(String, PathBuf), // Cannot read metadata
    FileWrite(String, PathBuf),   // Cannot write file
    Compression(String),          // Compression failed
    Directory(String, PathBuf),   // Cannot create directory
    Environment(String),          // Env var error
    PathConversion(PathBuf),      // Path to string failed
    Hash(String),                 // Hash computation failed
}
```

## Troubleshooting

### Common Issues

1. **Weights file not found**
   - Ensure `weights/model.safetensors` exists in project root
   - Check build script path resolution

2. **Build script not rerunning**
   - Verify `cargo:rerun-if-changed` directives
   - Clean build with `cargo clean`

3. **Cross-compilation linker errors**
   - Install required cross-compilation toolchain
   - Configure `.cargo/config.toml` for target

4. **Environment variable not set**
   - Check build script output for `cargo:rustc-env=` lines
   - Verify build script executed successfully

## References

- [Cargo Build Scripts](https://doc.rust-lang.org/cargo/reference/build-scripts.html)
- [Cargo Configuration](https://doc.rust-lang.org/cargo/reference/config.html)
- [Rust Platform Support](https://doc.rust-lang.org/rustc/platform-support.html)
- [Cross Toolchain](https://github.com/cross-rs/cross)
