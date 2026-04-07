#!/usr/bin/env python3
"""
Generate Python reference tensors for correctness testing.

This script hooks into the Sundial Python model and saves all intermediate
tensors to disk. These tensors are then used as "golden references" to
verify the Rust implementation.

Usage:
    python scripts/generate_reference.py \
        --input /tmp/test_input.npy \
        --output /tmp/python_reference/
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path

# Try to import Sundial model - adjust import path as needed
try:
    from sundial_model.modeling_sundial import SundialModel
    from transformers import AutoConfig
except ImportError as e:
    print(f"Warning: Could not import Sundial model: {e}")
    print("This script requires the Python Sundial implementation.")
    print("Make sure the Sundial Python package is installed.")
    exit(1)


def load_sundial_model(model_name="thuml/sundial-base-128m"):
    """Load the Sundial model from Hugging Face"""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = SundialModel(config)
    model.eval()
    return model


def make_hook(storage_dict, name):
    """Create a forward hook to capture intermediate outputs"""
    def hook(module, input, output):
        # Handle both tuple outputs and single tensor outputs
        if isinstance(output, tuple):
            storage_dict[name] = output[0].detach().cpu().numpy()
        else:
            storage_dict[name] = output.detach().cpu().numpy()
    return hook


def generate_reference(input_path, output_dir, model_name="thuml/sundial-base-128m"):
    """
    Generate Python reference tensors for all intermediate layers.
    
    Args:
        input_path: Path to input numpy file (shape: [batch, timesteps, features])
        output_dir: Directory to save reference tensors
        model_name: Hugging Face model name or path
    """
    print(f"Loading model: {model_name}")
    model = load_sundial_model(model_name)
    
    # Load input data
    print(f"Loading input from: {input_path}")
    input_data = np.load(input_path)
    input_tensor = torch.from_numpy(input_data).float()
    
    # Squeeze if needed (input may be [1, 2880, 1], we want [1, 2880])
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.squeeze(-1)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Create storage for intermediates
    intermediates = {}
    
    # Register hooks on key components
    print("Registering hooks...")
    
    # Patch embedding layer
    if hasattr(model, 'embed_layer'):
        model.embed_layer.register_forward_hook(
            make_hook(intermediates, 'patch_embed_output')
        )
    
    # All decoder layers
    if hasattr(model, 'layers'):
        for i in range(len(model.layers)):
            model.layers[i].register_forward_hook(
                make_hook(intermediates, f'layer_{i}_output')
            )
    
    # Final layer norm
    if hasattr(model, 'norm'):
        model.norm.register_forward_hook(
            make_hook(intermediates, 'transformer_output')
        )
    
    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output = model(input_tensor)
    
    # Save predictions - output is MoeModelOutputWithPast
    if hasattr(output, 'last_hidden_state'):
        intermediates['predictions'] = output.last_hidden_state.detach().cpu().numpy()
    elif isinstance(output, tuple):
        intermediates['predictions'] = output[0].detach().cpu().numpy()
    else:
        intermediates['predictions'] = output.detach().cpu().numpy()
    
    # Also save the input for reference
    intermediates['input'] = input_tensor.cpu().numpy()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save all intermediates
    print(f"\nSaving {len(intermediates)} tensors to: {output_path}")
    for name, tensor in intermediates.items():
        save_path = output_path / f"{name}.npy"
        np.save(save_path, tensor)
        print(f"  Saved {name:30s}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'input_shape': input_tensor.shape,
        'output_shape': output.last_hidden_state.shape if hasattr(output, 'last_hidden_state') else output.shape,
        'num_layers': len(model.layers) if hasattr(model, 'layers') else 0,
        'tensor_names': list(intermediates.keys()),
    }
    
    import json
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nReference generation complete!")
    print(f"Total tensors saved: {len(intermediates)}")
    return intermediates


def generate_rope_reference(input_path, output_dir, dim=64, max_position=1000, base=10000.0):
    """
    Generate RoPE-specific reference tensors.
    
    This creates isolated RoPE test cases with known inputs/outputs.
    """
    print(f"\nGenerating RoPE reference tensors...")
    
    # Create test inputs
    torch.manual_seed(42)
    batch_size = 2
    num_heads = 4
    seq_len = 10
    
    q_input = torch.randn(batch_size, num_heads, seq_len, dim)
    k_input = torch.randn(batch_size, num_heads, seq_len, dim)
    
    # Implement RoPE in PyTorch for reference
    def create_rope(dim, max_position, base):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position).float().unsqueeze(1)
        freqs = t * inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
    
    def apply_rope(x, cos, sin):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin
    
    cos, sin = create_rope(dim, max_position, base)
    
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
    }
    
    for name, tensor in tensors.items():
        np.save(output_path / f"{name}.npy", tensor)
        print(f"  Saved {name:30s}: shape={tensor.shape}")
    
    return tensors


def main():
    parser = argparse.ArgumentParser(description="Generate Python reference tensors")
    parser.add_argument('--input', type=str, default=None,
                        help='Input numpy file (optional, will generate random if not provided)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for reference tensors')
    parser.add_argument('--model', type=str, default="thuml/sundial-base-128m",
                        help='Hugging Face model name or path')
    parser.add_argument('--rope-only', action='store_true',
                        help='Only generate RoPE reference tensors')
    parser.add_argument('--full-model', action='store_true',
                        help='Generate full model intermediates')
    
    args = parser.parse_args()
    
    if args.rope_only:
        generate_rope_reference(args.input or "/tmp/test_input.npy", args.output)
    else:
        if args.input is None:
            # Generate random input for testing
            print("No input provided, generating random test input...")
            input_path = Path(args.output) / "input.npy"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            # Shape: [batch=1, timesteps=2880, features=1]
            random_input = np.random.randn(1, 2880, 1).astype(np.float32)
            np.save(input_path, random_input)
            args.input = str(input_path)
        
        if args.full_model:
            generate_reference(args.input, args.output, args.model)
        else:
            # Default: generate both RoPE and full model references
            rope_dir = os.path.join(args.output, "rope")
            os.makedirs(rope_dir, exist_ok=True)
            generate_rope_reference(args.input, rope_dir)
            
            if args.input:
                generate_reference(args.input, args.output, args.model)


if __name__ == '__main__':
    main()
