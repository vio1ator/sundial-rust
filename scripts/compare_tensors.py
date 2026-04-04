#!/usr/bin/env python3
"""
Compare tensors saved from Python and Rust implementations.

Usage:
    # Save tensors from both implementations
    python scripts/compare_intermediates.py --output /tmp/python_intermediates.npz
    SUNDIAL_DEBUG=1 cargo run --release --example compare_intermediates
    
    # Compare them
    python scripts/compare_tensors.py --python /tmp/python_intermediates.npz --rust /tmp/
"""

import argparse
import numpy as np
import os
import struct


def load_tensor_from_bin(filepath):
    """Load a tensor from the binary format saved by Rust."""
    with open(filepath, 'rb') as f:
        # Read dimension count
        dim_count = struct.unpack('<I', f.read(4))[0]
        
        # Read dimensions
        dims = []
        for _ in range(dim_count):
            dims.append(struct.unpack('<I', f.read(4))[0])
        
        # Read data
        data = []
        while True:
            try:
                val = struct.unpack('<f', f.read(4))[0]
                data.append(val)
            except struct.error:
                break
        
        return np.array(data, dtype=np.float32).reshape(dims)


def load_python_intermediates(filepath):
    """Load Python intermediates from .npz file."""
    data = np.load(filepath)
    return dict(data)


def compare_tensors(rust_tensor, python_tensor, name):
    """Compare two tensors and print statistics."""
    if rust_tensor.shape != python_tensor.shape:
        print(f"\n{name}:")
        print(f"  SHAPE MISMATCH: Rust {rust_tensor.shape} vs Python {python_tensor.shape}")
        return
    
    # Calculate differences
    diff = np.abs(rust_tensor - python_tensor)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    # Find largest differences
    largest_indices = np.unravel_index(np.argmax(diff), diff.shape)
    largest_diff = diff[largest_indices]
    
    print(f"\n{name}:")
    print(f"  Shape: {rust_tensor.shape}")
    print(f"  Max abs diff: {max_diff:.8f}")
    print(f"  Mean abs diff: {mean_diff:.8f}")
    print(f"  Std abs diff: {std_diff:.8f}")
    print(f"  Largest diff at index {largest_indices}: {largest_diff:.8f}")
    
    # Check if within tolerance
    if max_diff < 1e-4:
        print(f"  ✓ PASS: Within tolerance (1e-4)")
    elif max_diff < 1e-2:
        print(f"  ⚠ WARNING: Small differences detected")
    else:
        print(f"  ✗ FAIL: Large differences detected!")
    
    # Show some sample values
    print(f"  Sample Rust values:  {rust_tensor.flat[:min(5, rust_tensor.size)]}")
    print(f"  Sample Python values: {python_tensor.flat[:min(5, python_tensor.size)]}")


def find_rust_tensors(rust_dir):
    """Find all Rust tensor files in a directory."""
    tensors = {}
    for filename in os.listdir(rust_dir):
        if filename.endswith('_rust.bin'):
            name = filename.replace('_rust.bin', '')
            filepath = os.path.join(rust_dir, filename)
            try:
                tensors[name] = load_tensor_from_bin(filepath)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return tensors


def main():
    parser = argparse.ArgumentParser(description="Compare Python and Rust tensors")
    parser.add_argument("--python", type=str, required=True,
                        help="Path to Python .npz file")
    parser.add_argument("--rust", type=str, required=True,
                        help="Directory containing Rust .bin files")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Tolerance for comparison (default: 1e-4)")
    
    args = parser.parse_args()
    
    print("Loading Python intermediates...")
    python_tensors = load_python_intermediates(args.python)
    print(f"Loaded {len(python_tensors)} Python tensors")
    
    print("\nLoading Rust tensors...")
    rust_tensors = find_rust_tensors(args.rust)
    print(f"Loaded {len(rust_tensors)} Rust tensors")
    
    # Find common tensors
    common = set(python_tensors.keys()) & set(rust_tensors.keys())
    print(f"\nFound {len(common)} common tensors to compare")
    
    if not common:
        print("\nNo common tensors found!")
        print("Python tensors:", list(python_tensors.keys()))
        print("Rust tensors:", list(rust_tensors.keys()))
        return
    
    # Compare each common tensor
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name in sorted(common):
        python_tensor = python_tensors[name]
        rust_tensor = rust_tensors[name]
        
        compare_tensors(rust_tensor, python_tensor, name)
        
        max_diff = np.max(np.abs(rust_tensor - python_tensor))
        if max_diff < args.tolerance:
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(common)}")
    print(f"Failed: {failed}/{len(common)}")
    
    if failed > 0:
        print("\nSome comparisons failed. Check the output above for details.")
    else:
        print("\nAll comparisons passed!")


if __name__ == "__main__":
    main()
