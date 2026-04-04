#!/bin/bash
# Cross-compile Sundial for Linux x86_64 (GNU toolchain)
# Requires: gcc-multilib musl-tools

set -e

echo "=== Cross-compiling for Linux x86_64 ==="
echo "Target: x86_64-unknown-linux-gnu"

# Install dependencies if on Debian/Ubuntu
if command -v apt-get &> /dev/null; then
    echo "Checking dependencies..."
    if ! dpkg -l | grep -q gcc-multilib; then
        echo "Installing gcc-multilib..."
        sudo apt-get update
        sudo apt-get install -y gcc-multilib musl-tools
    fi
fi

# Install Rust target if needed
rustup target add x86_64-unknown-linux-gnu

# Build with cross or cargo
if command -v cross &> /dev/null; then
    echo "Using cross for cross-compilation..."
    cross build --target x86_64-unknown-linux-gnu --release --profile release-x86_64-linux
else
    echo "Using cargo with native cross-compilation..."
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc \
    cargo build --target x86_64-unknown-linux-gnu --release --profile release-x86_64-linux
fi

echo "=== Build complete ==="
echo "Binary location: target/x86_64-unknown-linux-gnu/release/sundial-rust"
