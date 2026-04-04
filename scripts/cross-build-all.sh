#!/bin/bash
# Convenience script to cross-compile for all supported targets

set -e

echo "=========================================="
echo "Sundial Cross-Compilation Script"
echo "=========================================="
echo ""

TARGETS=(
    "x86_64-unknown-linux-gnu:Linux x86_64"
    "aarch64-unknown-linux-gnu:Linux ARM64"
    "x86_64-pc-windows-msvc:Windows x86_64"
    "x86_64-apple-darwin:macOS x86_64"
    "aarch64-apple-darwin:macOS ARM64"
)

# Parse arguments
BUILD_ALL=true
BUILD_TARGETS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --linux-x86_64)
            BUILD_TARGETS+=("x86_64-unknown-linux-gnu")
            BUILD_ALL=false
            shift
            ;;
        --linux-aarch64)
            BUILD_TARGETS+=("aarch64-unknown-linux-gnu")
            BUILD_ALL=false
            shift
            ;;
        --windows)
            BUILD_TARGETS+=("x86_64-pc-windows-msvc")
            BUILD_ALL=false
            shift
            ;;
        --macos-intel)
            BUILD_TARGETS+=("x86_64-apple-darwin")
            BUILD_ALL=false
            shift
            ;;
        --macos-arm)
            BUILD_TARGETS+=("aarch64-apple-darwin")
            BUILD_ALL=false
            shift
            ;;
        --all)
            BUILD_ALL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--linux-x86_64|--linux-aarch64|--windows|--macos-intel|--macos-arm|--all]"
            exit 1
            ;;
    esac
done

# If building all, use all targets
if [ "$BUILD_ALL" = true ]; then
    for target_info in "${TARGETS[@]}"; do
        TARGET=${target_info%%:*}
        BUILD_TARGETS+=("$TARGET")
    done
fi

# Build each target
for target in "${BUILD_TARGETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Building: $target"
    echo "=========================================="
    
    # Add target to rustup if not already present
    rustup target add "$target" 2>/dev/null || true
    
    # Determine profile based on target
    case $target in
        *linux*gnu)
            if [[ "$target" == *"x86_64"* ]]; then
                PROFILE="release-x86_64-linux"
            else
                PROFILE="release-aarch64-linux"
            fi
            ;;
        *windows*msvc)
            PROFILE="release-x86_64-windows"
            ;;
        *apple*darwin)
            if [[ "$target" == *"x86_64"* ]]; then
                PROFILE="release-x86_64-macos"
            else
                PROFILE="release-aarch64-macos"
            fi
            ;;
        *)
            PROFILE="release"
            ;;
    esac
    
    echo "Using profile: $PROFILE"
    
    # Build
    if command -v cross &> /dev/null; then
        echo "Using cross..."
        cross build --target "$target" --release --profile "$PROFILE"
    else
        echo "Using cargo with native cross-compilation..."
        cargo build --target "$target" --release --profile "$PROFILE"
    fi
    
    echo "Successfully built: $target"
done

echo ""
echo "=========================================="
echo "All builds complete!"
echo "=========================================="
echo ""
echo "Binary locations:"
for target in "${BUILD_TARGETS[@]}"; do
    case $target in
        *windows*)
            echo "  $target: target/$target/release/sundial-rust.exe"
            ;;
        *)
            echo "  $target: target/$target/release/sundial-rust"
            ;;
    esac
done
