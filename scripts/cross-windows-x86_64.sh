#!/bin/bash
# Cross-compile Sundial for Windows x86_64 (MSVC toolchain)
# Requires: Windows SDK and Visual Studio Build Tools

set -e

echo "=== Cross-compiling for Windows x86_64 ==="
echo "Target: x86_64-pc-windows-msvc"

# Note: MSVC cross-compilation requires building on Windows
# or using a Windows build environment (e.g., GitHub Actions Windows runner)

# Install Rust target if needed
rustup target add x86_64-pc-windows-msvc

# Build using cargo
echo "Building for Windows MSVC..."
cargo build --target x86_64-pc-windows-msvc --release --profile release-x86_64-windows

echo "=== Build complete ==="
echo "Binary location: target/x86_64-pc-windows-msvc/release/sundial-rust.exe"
