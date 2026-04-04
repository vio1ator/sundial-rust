# Cross-Compilation Guide

This document describes how to cross-compile Sundial for different platforms from a single build environment.

## Supported Targets

| Target | Triple | Profile | Architecture |
|--------|--------|---------|--------------|
| Linux x86_64 | `x86_64-unknown-linux-gnu` | `release-x86_64-linux` | x86_64 |
| Linux ARM64 | `aarch64-unknown-linux-gnu` | `release-aarch64-linux` | ARM64 |
| Windows x86_64 | `x86_64-pc-windows-msvc` | `release-x86_64-windows` | x86_64 |
| macOS x86_64 | `x86_64-apple-darwin` | `release-x86_64-macos` | x86_64 |
| macOS ARM64 | `aarch64-apple-darwin` | `release-aarch64-macos` | ARM64 |

## Quick Start

### Using the Convenience Script

```bash
# Build all targets
./scripts/cross-build-all.sh

# Build specific targets
./scripts/cross-build-all.sh --linux-x86_64
./scripts/cross-build-all.sh --linux-aarch64
./scripts/cross-build-all.sh --windows
./scripts/cross-build-all.sh --macos-intel
./scripts/cross-build-all.sh --macos-arm
```

### Using Individual Scripts

```bash
# Linux x86_64
./scripts/cross-linux-x86_64.sh

# Linux ARM64
./scripts/cross-linux-aarch64.sh

# Windows x86_64
./scripts/cross-windows-x86_64.sh

# macOS x86_64
./scripts/cross-macos-x86_64.sh

# macOS ARM64
./scripts/cross-macos-aarch64.sh
```

## Manual Cross-Compilation

### Linux x86_64

**Prerequisites (on macOS host):**
```bash
brew install FiloSottile/musl-cross/musl-cross
```

**Prerequisites (on Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y gcc-multilib musl-tools
```

**Build command:**
```bash
rustup target add x86_64-unknown-linux-gnu
cargo build --target x86_64-unknown-linux-gnu --release --profile release-x86_64-linux
```

### Linux ARM64

**Prerequisites (on macOS host):**
```bash
brew install FiloSottile/musl-cross/musl-cross
```

**Prerequisites (on Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y gcc-aarch64-linux-gnu musl-tools
```

**Build command:**
```bash
rustup target add aarch64-unknown-linux-gnu
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
cargo build --target aarch64-unknown-linux-gnu --release --profile release-aarch64-linux
```

### Windows x86_64

**Prerequisites:**
- Windows SDK 10.0.19041.0 or later
- Visual Studio Build Tools with MSVC compiler
- Or use GitHub Actions Windows runner

**Build command (on Windows):**
```bash
rustup target add x86_64-pc-windows-msvc
cargo build --target x86_64-pc-windows-msvc --release --profile release-x86_64-windows
```

**Build command (on macOS/Linux with cross):**
```bash
rustup target add x86_64-pc-windows-msvc
cross build --target x86_64-pc-windows-msvc --release --profile release-x86_64-windows
```

### macOS x86_64

**Prerequisites:**
- Xcode Command Line Tools

**Build command:**
```bash
rustup target add x86_64-apple-darwin
CFLAGS_x86_64_apple_darwin="-mmacosx-version-min=10.13" \
cargo build --target x86_64-apple-darwin --release --profile release-x86_64-macos
```

### macOS ARM64

**Prerequisites:**
- Xcode Command Line Tools

**Build command:**
```bash
rustup target add aarch64-apple-darwin
CFLAGS_aarch64_apple_darwin="-mmacosx-version-min=11.0" \
cargo build --target aarch64-apple-darwin --release --profile release-aarch64-macos
```

## Using Cargo Cross

For more reliable cross-compilation, especially for Linux targets, consider using [`cargo-cross`](https://github.com/cross-rs/cross):

```bash
# Install cross
cargo install cross

# Build for any target
cross build --target <target-triple> --release --profile <profile-name>
```

## Configuration

### Cargo.toml Settings

Cross-compilation configuration is defined in `Cargo.toml` under `[package.metadata.cross.target.*]`:

```toml
[package.metadata.cross.target.x86_64-unknown-linux-gnu]
pre-build = ["apt-get update && apt-get install -y gcc-multilib musl-tools"]
rustflags = ["-C", "link-arg=-lm"]
```

### Build Profiles

Custom release profiles for each target are defined in `Cargo.toml`:

```toml
[profile.release-x86_64-linux]
inherits = "release"
opt-level = 3
lto = true
codegen-units = 1
```

## Platform Detection

The build script (`build.rs`) automatically detects the target platform and sets environment variables:

- `TARGET_OS`: Operating system (linux, macos, windows)
- `TARGET_ARCH`: Architecture (x86_64, aarch64)
- `TARGET_FAMILY`: Target family (unix, windows)

These can be used for conditional compilation:

```rust
#[cfg(target_os = "linux")]
// Linux-specific code
#[cfg(target_os = "windows")]
// Windows-specific code
#[cfg(target_os = "macos")]
// macOS-specific code
```

## Binary Output Locations

After building, binaries are located at:

| Target | Binary Path |
|--------|-------------|
| x86_64-unknown-linux-gnu | `target/x86_64-unknown-linux-gnu/release/sundial-rust` |
| aarch64-unknown-linux-gnu | `target/aarch64-unknown-linux-gnu/release/sundial-rust` |
| x86_64-pc-windows-msvc | `target/x86_64-pc-windows-msvc/release/sundial-rust.exe` |
| x86_64-apple-darwin | `target/x86_64-apple-darwin/release/sundial-rust` |
| aarch64-apple-darwin | `target/aarch64-apple-darwin/release/sundial-rust` |

## Troubleshooting

### Linker Not Found

If you get "linker not found" errors:

1. Install the appropriate cross-compilation toolchain
2. Set the linker environment variable:
   ```bash
   export CARGO_TARGET_<ARCH>_LINKER=<linker>
   ```

### Missing System Libraries

For Linux targets, ensure required development libraries are installed:
```bash
# Ubuntu/Debian
sudo apt-get install -y gcc-multilib musl-tools

# macOS
brew install FiloSottile/musl-cross/musl-cross
```

### Platform-Specific Compilation Errors

Use conditional compilation to handle platform differences:

```rust
#[cfg(target_os = "linux")]
use libc::some_function();

#[cfg(target_os = "windows")]
use windows::some_function();

#[cfg(target_os = "macos")]
use core_foundation::some_function();
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Cross-Compile

on: [push, pull_request]

jobs:
  build:
    strategy:
      matrix:
        target:
          - x86_64-unknown-linux-gnu
          - aarch64-unknown-linux-gnu
          - x86_64-pc-windows-msvc
          - x86_64-apple-darwin
          - aarch64-apple-darwin

    runs-on: ${{ matrix.target == 'x86_64-pc-windows-msvc' && 'windows-latest' || 
                  (matrix.target == 'x86_64-apple-darwin' || matrix.target == 'aarch64-apple-darwin' && 'macos-latest' || 'ubuntu-latest') }}

    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: ${{ matrix.target }}
          override: true

      - name: Build
        run: cargo build --target ${{ matrix.target }} --release --profile release-${{ matrix.target }}
```

## Best Practices

1. **Use `cargo-cross`**: For Linux targets, `cross` provides containerized builds with all dependencies
2. **Test on target**: Always test cross-compiled binaries on the target platform
3. **Use CI/CD**: Automate cross-compilation in your CI/CD pipeline
4. **Document dependencies**: Clearly document system dependencies for each target
5. **Version compatibility**: Ensure minimum OS versions are appropriate for your target audience
