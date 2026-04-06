#!/usr/bin/env python3
"""
Standalone RoPE reference generator - does not require Sundial Python package.
Generates RoPE reference tensors for correctness testing.
"""

import os
import numpy as np
import torch
from pathlib import Path


def create_rope_embeddings(dim: int, max_position: int, base: float = 10000.0):
    """
    Create RoPE embeddings (cos and sin matrices).
    
    Args:
        dim: Dimension of the embedding (should be even)
        max_position: Maximum position to generate embeddings for
        base: Base value for frequency calculation (default 10000)
    
    Returns:
        cos: Cosine embeddings of shape (max_position, dim)
        sin: Sine embeddings of shape (max_position, dim)
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_position).float().unsqueeze(1)
    freqs = t * inv_freq.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply RoPE to input tensor.
    
    Args:
        x: Input tensor of shape (batch, heads, seq_len, dim)
        cos: Cosine embeddings of shape (max_position, dim)
        sin: Sine embeddings of shape (max_position, dim)
    
    Returns:
        RoPE-applied tensor of same shape as x
    """
    seq_len = x.shape[-2]
    
    # Slice cos/sin to match the input sequence length
    # cos/sin shape: (max_position, dim) -> (seq_len, dim)
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    
    # Reshape to match x's dimensions
    # cos/sin shape: (seq_len, dim) -> (1, 1, seq_len, dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def generate_rope_reference(output_dir: str, dim: int = 64, max_position: int = 1000, base: float = 10000.0, seed: int = 42):
    """
    Generate RoPE reference tensors.
    
    Args:
        output_dir: Directory to save reference tensors
        dim: Dimension of RoPE embeddings
        max_position: Maximum position
        base: Base value for frequency calculation
        seed: Random seed for reproducibility
    """
    print(f"\nGenerating RoPE reference tensors...")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create test inputs
    batch_size = 2
    num_heads = 4
    seq_len = 10
    
    q_input = torch.randn(batch_size, num_heads, seq_len, dim)
    k_input = torch.randn(batch_size, num_heads, seq_len, dim)
    
    # Create RoPE embeddings
    cos, sin = create_rope_embeddings(dim, max_position, base)
    
    # Apply RoPE
    q_output = apply_rope(q_input, cos, sin)
    k_output = apply_rope(k_input, cos, sin)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tensors = {
        'rope_q_input': q_input.numpy(),
        'rope_k_input': k_input.numpy(),
        'rope_q_output': q_output.numpy(),
        'rope_k_output': k_output.numpy(),
        'rope_cos': cos.numpy(),
        'rope_sin': sin.numpy(),
    }
    
    for name, tensor in tensors.items():
        np.save(output_path / f"{name}.npy", tensor)
        print(f"  Saved {name:30s}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    # Save metadata
    metadata = {
        'dim': dim,
        'max_position': max_position,
        'base': base,
        'batch_size': batch_size,
        'num_heads': num_heads,
        'seq_len': seq_len,
        'tensor_names': list(tensors.keys()),
    }
    
    import json
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nRoPE reference generation complete!")
    print(f"Total tensors saved: {len(tensors)}")
    return tensors


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate RoPE reference tensors (standalone)")
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for reference tensors')
    parser.add_argument('--dim', type=int, default=64,
                        help='Dimension of RoPE embeddings')
    parser.add_argument('--max-position', type=int, default=1000,
                        help='Maximum position for RoPE')
    parser.add_argument('--base', type=float, default=10000.0,
                        help='Base value for frequency calculation')
    
    args = parser.parse_args()
    
    generate_rope_reference(args.output, args.dim, args.max_position, args.base)


if __name__ == '__main__':
    main()
