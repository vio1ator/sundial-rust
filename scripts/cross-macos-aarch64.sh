#!/bin/bash
# Cross-compile Sundial for macOS ARM64 (Apple Silicon)
# Requires: Xcode Command Line Tools

set -e

echo "=== Cross-compiling for macOS ARM64 ==="
echo "Target: aarch64-apple-darwin"

# Install Rust target if needed
rustup target add aarch64-apple-darwin

# Build using cargo
echo "Building for macOS ARM64..."
CFLAGS_aarch64_apple_darwin="-mmacosx-version-min=11.0" \
cargo build --target aarch64-apple-darwin --release --profile release-aarch64-macos

echo "=== Build complete ==="
echo "Binary location: target/aarch64-apple-darwin/release/sundial-rust"
