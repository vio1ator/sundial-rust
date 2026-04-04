# Build Requirements and Setup Guide

This document provides comprehensive instructions for setting up the development environment and building the Sundial Rust project.

## 📋 Table of Contents

- [Rust Toolchain Requirements](#rust-toolchain-requirements)
- [Platform-Specific Dependencies](#platform-specific-dependencies)
- [Setup Instructions](#setup-instructions)
- [Build Commands](#build-commands)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Cross-Compilation](#cross-compilation)

---

## Rust Toolchain Requirements

### Required Version

- **Rust**: 1.70.0 or higher
- **Edition**: 2021
- **Recommended**: Latest stable version

### Installation

#### macOS (Homebrew)
```bash
brew install rustup-init
rustup-init
```

#### Linux (apt-get)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

#### Windows
Download `rustup-init.exe` from [rustup.rs](https://rustup.rs/) and run the installer.

### Verify Installation
```bash
rustc --version  # Should show 1.70.0 or higher
cargo --version  # Should match rustc version
```

---

## Platform-Specific Dependencies

### Linux (x86_64 and aarch64)

#### Required System Libraries
```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    gcc \
    musl-tools \
    pkg-config \
    libssl-dev

# For x86_64 cross-compilation
sudo apt-get install -y gcc-multilib

# For aarch64 cross-compilation
sudo apt-get install -y gcc-aarch64-linux-gnu
```

#### Required Rust Targets
```bash
rustup target add x86_64-unknown-linux-gnu
rustup target add aarch64-unknown-linux-gnu
```

### macOS (x86_64 and aarch64)

#### Required Tools
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify installation
clang --version
```

#### Required Rust Targets
```bash
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin
```

### Windows (x86_64)

#### Required Tools
- **Visual Studio Studio Build Tools** (2019 or later)
  - Download from [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
  - Select "Desktop development with C++" workload
  - Includes Windows SDK and MSVC compiler

#### Required Rust Targets
```powershell
rustup target add x86_64-pc-windows-msvc
```

#### Environment Variables (if using Windows SDK)
```powershell
# Set Windows SDK library directory
$env:WINSDK_LIB_DIR = "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\um\x64"
```

---

## Setup Instructions

### Quick Setup (Native Build)

1. **Install Rust Toolchain**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Install Platform Dependencies** (see Platform-Specific Dependencies above)

3. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/sundial-rust.git
   cd sundial-rust
   ```

4. **Download Model Weights**
   ```bash
   # Place model weights in weights/ directory
   # Required files:
   # - weights/model.safetensors (490MB)
   # - weights/config.json
   ```

5. **Build**
   ```bash
   cargo build --release
   ```

### Development Setup

For development with testing and linting:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install platform dependencies (see above)

# Clone and setup
git clone https://github.com/your-org/sundial-rust.git
cd sundial-rust

# Verify installation
cargo --version
rustc --version

# Run tests
cargo test

# Run linting
cargo clippy

# Check formatting
cargo fmt --check
```

---

## Build Commands

### Standard Build

```bash
# Debug build (faster compilation)
cargo build

# Release build (optimized)
cargo build --release
```

### Build with Specific Profile

```bash
# Linux x86_64 optimized
cargo build --release --profile release-x86_64-linux

# Linux ARM64 optimized
cargo build --release --profile release-aarch64-linux

# Windows x86_64 optimized
cargo build --release --profile release-x86_64-windows

# macOS x86_64 optimized
cargo build --release --profile release-x86_64-macos

# macOS ARM64 optimized
cargo build --release --profile release-aarch64-macos
```

### Using Build Scripts

#### Unix (build.sh)
```bash
# Native build
./build.sh

# Specific target
./build.sh --target x86_64-linux --profile release-x86_64-linux

# Clean build
./build.sh --clean

# Show help
./build.sh --help
```

#### Windows (build.bat)
```powershell
# Native build
build.bat

# Specific target
build.bat --target x86_64-windows --profile release-x86_64-windows

# Clean build
build.bat --clean

# Show help
build.bat --help
```

### Quality Checks

```bash
# Run all tests
cargo test

# Linting with clippy
cargo clippy

# Check formatting
cargo fmt --check

# All checks together
cargo test && cargo clippy && cargo fmt --check
```

---

## Troubleshooting Guide

### Common Build Issues

#### 1. Rust Toolchain Version Too Old

**Error:**
```
error: rustc 1.70.0 is required but found 1.65.0
```

**Solution:**
```bash
rustup update stable
rustup default stable
```

#### 2. Missing Platform Dependencies (Linux)

**Error:**
```
fatal error: openssl/opensslv.h: No such file or directory
```

**Solution:**
```bash
# Debian/Ubuntu
sudo apt-get install libssl-dev pkg-config

# Fedora/RHEL
sudo dnf install openssl-devel pkg-config
```

#### 3. MSVC Compiler Not Found (Windows)

**Error:**
```
error: failed to run custom build command for XXX
note: Could not find MSVC toolchain
```

**Solution:**
- Install Visual Studio Build Tools with "Desktop development with C++" workload
- Ensure Windows 10 SDK is installed
- Run `vcvars64.bat` before building (from VS Developer Command Prompt)

#### 4. Xcode Command Line Tools Missing (macOS)

**Error:**
```
xcrun: error: invalid active developer path
```

**Solution:**
```bash
xcode-select --install
```

#### 5. Missing Rust Target

**Error:**
```
error: toolchain 'stable-x86_64-unknown-linux-gnu' does not contain target 'aarch64-unknown-linux-gnu'
```

**Solution:**
```bash
rustup target add aarch64-unknown-linux-gnu
```

#### 6. Weights File Not Found

**Error:**
```
BuildError::MissingWeights: weights/model.safetensors
```

**Solution:**
```bash
# Download the model weights
# Place in weights/ directory
ls -lh weights/model.safetensors  # Should be ~490MB
```

#### 7. Cross-Compilation Failed

**Error:**
```
error: linking with `gcc` failed
```

**Solution:**
- Install cross-compilation toolchain:
  ```bash
  # Linux x86_64
  sudo apt-get install gcc-multilib musl-tools

  # Linux aarch64
  sudo apt-get install gcc-aarch64-linux-gnu musl-tools
  ```

- Use cargo-cross for containerized builds:
  ```bash
  cargo install cross
  cross build --target x86_64-unknown-linux-gnu --release
  ```

#### 8. Feature Flag Conflicts

**Error:**
```
error: feature `candle-cuda` requires feature `cuda`
```

**Solution:**
```bash
# Install CUDA toolkit if using CUDA
# Then build with:
cargo build --features candle-cuda
```

#### 9. Memory Issues During Build

**Error:**
```
Killed process (out of memory)
```

**Solution:**
- Reduce codegen units in profile:
  ```toml
  [profile.release]
  codegen-units = 16
  ```
- Or use Thin LTO instead of full LTO:
  ```toml
  lto = "thin"
  ```

#### 10. OpenSSL Linking Issues (Linux)

**Error:**
```
error: could not find OpenSSL headers
```

**Solution:**
```bash
# Install development headers
sudo apt-get install libssl-dev

# Or use vendored OpenSSL
cargo build --features vendored-openssl
```

---

## Cross-Compilation

For cross-compilation, refer to the [CROSS_COMPILATION.md](CROSS_COMPILATION.md) documentation.

### Quick Reference

```bash
# Install cargo-cross
cargo install cross

# Build for Linux x86_64 from macOS
cross build --target x86_64-unknown-linux-gnu --release

# Build for Linux aarch64 from macOS
cross build --target aarch64-unknown-linux-gnu --release

# Build for Windows from macOS (requires Windows host or CI)
cross build --target x86_64-pc-windows-msvc --release
```

### Supported Targets

| Target | Triple | Toolchain | Profile |
|--------|--------|-----------|---------|
| Linux x86_64 | `x86_64-unknown-linux-gnu` | GNU | `release-x86_64-linux` |
| Linux aarch64 | `aarch64-unknown-linux-gnu` | GNU | `release-aarch64-linux` |
| macOS x86_64 | `x86_64-apple-darwin` | Xcode | `release-x86_64-macos` |
| macOS aarch64 | `aarch64-apple-darwin` | Xcode | `release-aarch64-macos` |
| Windows x86_64 | `x86_64-pc-windows-msvc` | MSVC | `release-x86_64-windows` |

---

## Build Environment Variables

The build script sets the following environment variables for conditional compilation:

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `TARGET_OS` | Target operating system | `linux`, `macos`, `windows` |
| `TARGET_ARCH` | Target architecture | `x86_64`, `aarch64` |
| `TARGET_FAMILY` | Target family | `unix`, `windows` |
| `WEIGHTS_PATH` | Path to compressed weights | `/path/to/model.safetensors.gz` |
| `MODEL_SHA256` | SHA256 hash of weights | `a1b2c3d4...` |

---

## Build Output

After a successful build, binaries are located at:

```
target/
├── debug/           # Debug build
│   └── sundial-rust
└── release/         # Release build
    └── sundial-rust
```

For cross-compilation:
```
target/
├── x86_64-unknown-linux-gnu/
│   └── release/
│       └── sundial-rust
├── aarch64-unknown-linux-gnu/
│   └── release/
│       └── sundial-rust
└── x86_64-pc-windows-msvc/
    └── release/
        └── sundial-rust.exe
```

---

## Continuous Integration

For CI/CD setup, the following steps are recommended:

```yaml
# Example GitHub Actions
steps:
  - uses: actions/checkout@v3
  - uses: dtolnay/rust-toolchain@stable
    with:
      targets: x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu
  
  - name: Install dependencies (Linux)
    run: |
      sudo apt-get update
      sudo apt-get install -y gcc-multilib musl-tools
  
  - name: Build
    run: cargo build --release --profile release-x86_64-linux
  
  - name: Test
    run: cargo test
  
  - name: Lint
    run: cargo clippy
  
  - name: Format Check
    run: cargo fmt --check
```

---

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues and discussions
- Review the [README.md](README.md) for usage examples
- Consult [CROSS_COMPILATION.md](CROSS_COMPILATION.md) for cross-platform builds
