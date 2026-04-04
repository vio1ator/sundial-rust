# Sundial Rust - Cross-Platform Build Specification

## Overview

This document specifies the requirements and implementation plan for enabling cross-platform builds of the Sundial Rust time series forecasting model.

## Goals

1. **Enable cross-compilation** from macOS to Linux (x86_64, aarch64) and Windows (x86_64, aarch64)
2. **Maintain platform compatibility** for ML/Deep Learning dependencies (Candle, Polars)
3. **Produce statically-linked binaries** where possible for easier deployment
4. **Automate the build process** through CI/CD pipelines
5. **Ensure consistent behavior** across all supported platforms

## Supported Platforms

### Primary Targets

| Platform | Target Triple | Status | Notes |
|----------|---------------|--------|-------|
| macOS (Apple Silicon) | `aarch64-apple-darwin` | ✅ Native | Development platform |
| macOS (Intel) | `x86_64-apple-darwin` | ✅ Native | |
| Linux (x86_64) | `x86_64-unknown-linux-gnu` | 🔄 Cross | glibc-based |
| Linux (x86_64) | `x86_64-unknown-linux-musl` | 🔄 Cross | Static binary |
| Linux (ARM64) | `aarch64-unknown-linux-gnu` | 🔄 Cross | ARM servers |
| Windows (x86_64) | `x86_64-pc-windows-msvc` | 🔄 Cross | |

### Secondary Targets (Future)

| Platform | Target Triple | Status | Notes |
|----------|---------------|--------|-------|
| Linux (ARMv7) | `arm-unknown-linux-gnueabihf` | ⏳ Planned | Raspberry Pi |
| FreeBSD | `x86_64-unknown-freebsd` | ⏳ Planned | |

## Technical Requirements

### Dependencies Compatibility

| Dependency | Linux | Windows | Notes |
|------------|-------|---------|-------|
| candle-core | ✅ | ✅ | Pure Rust |
| candle-nn | ✅ | ✅ | Pure Rust |
| candle-transformers | ✅ | ✅ | Pure Rust |
| polars | ⚠️ | ⚠️ | Native components, may need feature flags |
| tokio | ✅ | ✅ | Pure Rust |
| reqwest | ✅ | ✅ | Needs platform-specific TLS |
| flate2 | ✅ | ✅ | Pure Rust |
| sha2 | ✅ | ✅ | Pure Rust |
| tempfile | ✅ | ✅ | Pure Rust |

### Platform-Specific Code

```rust
// Current platform guards in codebase
#[cfg(target_family = "unix")]  // Unix-specific code
#[cfg(target_family = "windows")]  // Windows-specific code
#[cfg(unix)]  // Unix-only code (permissions)
```

### Required Changes

1. **Weight extraction** (`src/weights/loader.rs`):
   - Already handles Unix permissions with `#[cfg(unix)]`
   - Need to verify Windows compatibility

2. **HTTP client** (`src/model/loader.rs`):
   - Already uses curl (Unix) and PowerShell (Windows)
   - Consider using `ureq` or `reqwest` for pure Rust HTTP

3. **Path handling**:
   - Use `std::path::PathBuf` consistently
   - Avoid hardcoded path separators

## Build Configuration

### Cargo Features

```toml
[features]
default = ["default-tls"]
# TLS backends
native-tls = ["reqwest/native-tls"]
rustls-tls = ["reqwest/rustls-tls"]
# Platform-specific
unix-permissions = []  # Enable Unix-specific permission handling
```

### Target-Specific Dependencies

```toml
[target.'cfg(unix)'.dependencies]
# Unix-specific dependencies

[target.'cfg(windows)'.dependencies]
# Windows-specific dependencies
```

## Cross-Compilation Setup

### Option 1: Using `cross` (Recommended)

```bash
# Install cross
cargo install cross

# Build for Linux (glibc)
cross build --target x86_64-unknown-linux-gnu --release

# Build for Linux (musl, static)
cross build --target x86_64-unknown-linux-musl --release

# Build for Windows
cross build --target x86_64-pc-windows-msvc --release
```

**Advantages:**
- Docker-based, isolated toolchains
- No manual linker setup required
- Consistent builds across environments

### Option 2: Manual Toolchain Setup

#### Linux (glibc)

```bash
# Install target
rustup target add x86_64-unknown-linux-gnu

# Install GNU linker (macOS)
brew install FiloSottile/musl-cross/musl-cross

# Create .cargo/config.toml
cat > .cargo/config.toml << 'EOF'
[target.x86_64-unknown-linux-gnu]
linker = "x86_64-linux-gnu-gcc"
EOF

# Build
cargo build --target x86_64-unknown-linux-gnu --release
```

#### Linux (musl, static)

```bash
# Install musl toolchain
brew install FiloSottile/musl-cross

# Create .cargo/config.toml
cat > .cargo/config.toml << 'EOF'
[target.x86_64-unknown-linux-musl]
linker = "x86_64-linux-musl-gcc"
rustflags = ["-C", "link-arg=-static"]
EOF

# Build
cargo build --target x86_64-unknown-linux-musl --release
```

#### Windows

```bash
# Add target
rustup target add x86_64-pc-windows-msvc

# Note: Native MSVC toolchain required
# Use cross for easier cross-compilation
cross build --target x86_64-pc-windows-msvc --release
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/cross-build.yml
name: Cross-Platform Builds

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  release:
    types: [published]

jobs:
  build-linux-gnu:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-action@stable
      - name: Build Linux (glibc)
        run: |
          cross build --target x86_64-unknown-linux-gnu --release
          cp target/x86_64-unknown-linux-gnu/release/sundial sundial-x86_64-linux

  build-linux-musl:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-action@stable
      - name: Build Linux (musl)
        run: |
          cross build --target x86_64-unknown-linux-musl --release
          cp target/x86_64-unknown-linux-musl/release/sundial sundial-x86_64-linux-musl

  build-windows:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-action@stable
      - name: Build Windows
        run: |
          cross build --target x86_64-pc-windows-msvc --release
          cp target/x86_64-pc-windows-msvc/release/sundial.exe sundial-x86_64-windows.exe

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-action@stable
      - name: Build macOS (aarch64)
        run: |
          cargo build --target aarch64-apple-darwin --release
          cp target/aarch64-apple-darwin/release/sundial sundial-aarch64-macos

      - name: Build macOS (x86_64)
        run: |
          cargo build --target x86_64-apple-darwin --release
          cp target/x86_64-apple-darwin/release/sundial sundial-x86_64-macos

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-action@stable
      - name: Run tests
        run: cargo test --release
```

## Testing Strategy

### Platform-Specific Tests

```rust
#[cfg(test)]
mod cross_platform_tests {
    #[cfg(unix)]
    #[test]
    fn test_unix_permissions() {
        // Test Unix-specific permission handling
    }

    #[cfg(windows)]
    #[test]
    fn test_windows_path_handling() {
        // Test Windows-specific path handling
    }

    #[test]
    fn test_platform_independent() {
        // Tests that work on all platforms
    }
}
```

### Integration Tests

1. **Binary verification**: Check binary exists and is executable
2. **Help command**: Run `sundial --help` on each platform
3. **Weight extraction**: Test embedded weight extraction
4. **Model loading**: Verify model loads correctly

### Test Matrix

| Platform | Binary | Tests | Integration |
|----------|--------|-------|-------------|
| macOS aarch64 | ✅ | ✅ | ✅ |
| macOS x86_64 | ✅ | ✅ | ✅ |
| Linux x86_64 (glibc) | ✅ | ✅ | ✅ |
| Linux x86_64 (musl) | ✅ | ✅ | ✅ |
| Windows x86_64 | ✅ | ✅ | ✅ |

## Known Issues & Limitations

### Current Limitations

1. **Polars native components**: May require platform-specific dependencies
2. **Candle ML backends**: Some backends may not be available on all platforms
3. **Weight extraction**: Temporary directory paths differ by platform
4. **File permissions**: Unix-specific permissions not applicable on Windows

### Mitigation Strategies

1. Use Cargo feature flags for platform-specific code
2. Provide fallback implementations for missing features
3. Document platform-specific requirements clearly
4. Use conditional compilation extensively

## Release Artifacts

### Expected Output

For each release, produce:

```
sundial-{version}-{platform}-{arch}.{ext}
├── sundial (or sundial.exe)
├── README.md
└── LICENSE
```

### Naming Convention

| Platform | Binary Name | Extension |
|----------|-------------|-----------|
| macOS aarch64 | `sundial-aarch64-macos` | (none) |
| macOS x86_64 | `sundial-x86_64-macos` | (none) |
| Linux x86_64 (glibc) | `sundial-x86_64-linux` | (none) |
| Linux x86_64 (musl) | `sundial-x86_64-linux-musl` | (none) |
| Windows x86_64 | `sundial-x86_64-windows` | `.exe` |

## Implementation Timeline

### Phase 1: Foundation (Week 1)
- [ ] Add `cross` to development workflow
- [ ] Update `.cargo/config.toml` for cross-compilation
- [ ] Fix any platform-specific code issues
- [ ] Test builds for Linux (glibc) and Windows

### Phase 2: CI/CD (Week 2)
- [ ] Create GitHub Actions workflow
- [ ] Configure build matrix for all targets
- [ ] Add artifact upload for releases
- [ ] Test CI pipeline end-to-end

### Phase 3: Testing (Week 3)
- [ ] Add platform-specific unit tests
- [ ] Create integration tests for each platform
- [ ] Test weight extraction on all platforms
- [ ] Document known issues and workarounds

### Phase 4: Release (Week 4)
- [ ] Create first cross-platform release
- [ ] Publish binaries to GitHub Releases
- [ ] Update documentation with platform support info
- [ ] Gather feedback and iterate

## Success Criteria

- [ ] All 6 primary targets build successfully
- [ ] CI pipeline runs on every commit
- [ ] Release artifacts available for all platforms
- [ ] Tests pass on all supported platforms
- [ ] Documentation covers platform-specific usage

## References

- [Rust Cross-Compilation Guide](https://doc.rust-lang.org/rustc/platform-support.html)
- [Cross Toolchain](https://github.com/cross-rs/cross)
- [Candle Platform Support](https://github.com/huggingface/candle)
- [Polars Platform Support](https://github.com/pola-rs/polars)

---

*Last updated: 2026-04-03*
*Status: Draft - Ready for implementation*
