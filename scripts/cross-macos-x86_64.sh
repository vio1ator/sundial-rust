#!/bin/bash
# Cross-compile Sundial for macOS x86_64 (Intel Mac)
# Requires: Xcode Command Line Tools

set -e

echo "=== Cross-compiling for macOS x86_64 ==="
echo "Target: x86_64-apple-darwin"

# Install Rust target if needed
rustup target add x86_64-apple-darwin

# Build using cargo
echo "Building for macOS x86_64..."
CFLAGS_x86_64_apple_darwin="-mmacosx-version-min=10.13" \
cargo build --target x86_64-apple-darwin --release --profile release-x86_64-macos

echo "=== Build complete ==="
echo "Binary location: target/x86_64-apple-darwin/release/sundial-rust"
