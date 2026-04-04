#!/bin/bash
# Script to download Sundial model weights from HuggingFace

set -e

MODEL_REPO="thuml/sundial-base-128m"
WEIGHTS_DIR="./weights"
MODEL_FILE="model.safetensors"
CONFIG_FILE="config.json"

echo "=== Sundial Model Weights Downloader ==="
echo "Model: $MODEL_REPO"
echo "Target directory: $WEIGHTS_DIR"
echo ""

# Create weights directory if it doesn't exist
mkdir -p "$WEIGHTS_DIR"

# Check if huggingface-cli is available
if command -v huggingface-cli &> /dev/null; then
    echo "[1] Using huggingface-cli to download weights..."
    huggingface-cli download "$MODEL_REPO" --local-dir "$WEIGHTS_DIR"
    echo "Download complete!"
else
    echo "[1] huggingface-cli not found, using manual download with curl..."
    
    echo "Downloading model.safetensors (this may take a few minutes)..."
    curl -L "https://huggingface.co/$MODEL_REPO/resolve/main/$MODEL_FILE" -o "$WEIGHTS_DIR/$MODEL_FILE"
    
    echo "Downloading config.json..."
    curl -L "https://huggingface.co/$MODEL_REPO/resolve/main/$CONFIG_FILE" -o "$WEIGHTS_DIR/$CONFIG_FILE"
    
    echo "Download complete!"
fi

echo ""
echo "=== Verification ==="
if [ -f "$WEIGHTS_DIR/$MODEL_FILE" ]; then
    echo "✓ model.safetensors exists ($(ls -lh $WEIGHTS_DIR/$MODEL_FILE | awk '{print $5}'))"
else
    echo "✗ model.safetensors not found!"
    exit 1
fi

if [ -f "$WEIGHTS_DIR/$CONFIG_FILE" ]; then
    echo "✓ config.json exists"
else
    echo "✗ config.json not found!"
    exit 1
fi

echo ""
echo "Model weights downloaded successfully to $WEIGHTS_DIR/"