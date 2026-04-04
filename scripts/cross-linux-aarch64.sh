#!/bin/bash
# Cross-compile Sundial for Linux ARM64 (GNU toolchain)
# Requires: gcc-aarch64-linux-gnu musl-tools

set -e

echo "=== Cross-compiling for Linux ARM64 ==="
echo "Target: aarch64-unknown-linux-gnu"

# Install dependencies if on Debian/Ubuntu
if command -v apt-get &> /dev/null; then
    echo "Checking dependencies..."
    if ! dpkg -l | grep -q gcc-aarch64-linux-gnu; then
        echo "Installing gcc-aarch64-linux-gnu..."
        sudo apt-get update
        sudo apt-get install -y gcc-aarch64-linux-gnu musl-tools
    fi
fi

# Install Rust target if needed
rustup target add aarch64-unknown-linux-gnu

# Build with cross or cargo
if command -v cross &> /dev/null; then
    echo "Using cross for cross-compilation..."
    cross build --target aarch64-unknown-linux-gnu --release --profile release-aarch64-linux
else
    echo "Using cargo with native cross-compilation..."
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    cargo build --target aarch64-unknown-linux-gnu --release --profile release-aarch64-linux
fi

echo "=== Build complete ==="
echo "Binary location: target/aarch64-unknown-linux-gnu/release/sundial-rust"
