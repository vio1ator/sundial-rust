#!/bin/bash
# Build script for Sundial Rust on Unix-like systems
# Usage: ./build.sh [OPTIONS]
#
# Options:
#   --target <TRIPLE>    Target platform (e.g., x86_64-unknown-linux-gnu, aarch64-apple-darwin)
#   --profile <PROFILE>  Build profile (default: release)
#   --clean              Clean build artifacts before building
#   --help               Show this help message
#
# Examples:
#   ./build.sh                          # Build for current platform
#   ./build.sh --target x86_64-unknown-linux-gnu
#   ./build.sh --profile release --clean

set -e

# Default values
TARGET=""
PROFILE="release"
CLEAN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print error message and exit
error() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Print warning message
warn() {
    echo -e "${YELLOW}Warning: $1${NC}" >&2
}

# Print info message
info() {
    echo -e "${GREEN}$1${NC}"
}

# Show help
show_help() {
    cat << 'EOF'
Build script for Sundial Rust on Unix-like systems

Usage: ./build.sh [OPTIONS]

Options:
  --target <TRIPLE>    Target platform (e.g., x86_64-unknown-linux-gnu, aarch64-apple-darwin)
  --profile <PROFILE>  Build profile (default: release)
  --clean              Clean build artifacts before building
  --help               Show this help message

Examples:
  ./build.sh                          # Build for current platform
  ./build.sh --target x86_64-unknown-linux-gnu
  ./build.sh --profile release --clean

Profiles available:
  release                    Standard release build
  release-x86_64-linux       Linux x86_64 with full LTO
  release-aarch64-linux      Linux ARM64 with Thin LTO
  release-x86_64-windows     Windows x86_64 with MSVC
  release-aarch64-macos      macOS ARM64
  release-x86_64-macos       macOS x86_64
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate profile
if ! grep -q "^\[profile.$PROFILE\]" Cargo.toml 2>/dev/null; then
    warn "Profile '$PROFILE' not found in Cargo.toml, using default release settings"
fi

# Clean if requested
if [ "$CLEAN" = true ]; then
    info "Cleaning build artifacts..."
    cargo clean
fi

# Add target if specified
if [ -n "$TARGET" ]; then
    info "Adding Rust target: $TARGET"
    rustup target add "$TARGET" || error "Failed to add target: $TARGET"
fi

# Build command
BUILD_CMD="cargo build"

if [ -n "$TARGET" ]; then
    BUILD_CMD="$BUILD_CMD --target $TARGET"
fi

BUILD_CMD="$BUILD_CMD --profile $PROFILE"

info "Running: $BUILD_CMD"
echo ""

# Execute build
eval "$BUILD_CMD"

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo ""
    info "Build successful!"
    
    # Show binary location
    if [ -n "$TARGET" ]; then
        BINARY_PATH="target/$TARGET/$PROFILE/sundial-rust"
    else
        BINARY_PATH="target/$PROFILE/sundial-rust"
    fi
    
    # Add .exe for Windows targets
    if [[ "$TARGET" == *"windows"* ]]; then
        BINARY_PATH="$BINARY_PATH.exe"
    fi
    
    if [ -f "$BINARY_PATH" ]; then
        echo "Binary location: $BINARY_PATH"
    fi
else
    error "Build failed"
fi
