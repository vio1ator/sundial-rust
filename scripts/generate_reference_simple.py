#!/usr/bin/env python3
"""
Generate Python reference tensors for correctness testing.
Simplified version that uses transformers AutoModelForCausalLM directly.
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM

def load_sundial_model(model_name="thuml/sundial-base-128m"):
    """Load the Sundial model from Hugging Face"""
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return model


def generate_full_model_reference(input_path, output_dir, model_name="thuml/sundial-base-128m"):
    """
    Generate full model reference tensors.
    
    Args:
        input_path: Path to input numpy file (shape: [batch, timesteps, features])
        output_dir: Directory to save reference tensors
        model_name: Hugging Face model name
    """
    model = load_sundial_model(model_name)
    
    # Load input data
    print(f"Loading input from: {input_path}")
    input_data = np.load(input_path)
    input_tensor = torch.from_numpy(input_data).float()
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Run forward pass - model expects [batch_size, seq_len]
    print("Running forward pass...")
    # Squeeze last dimension if needed (model expects 2D: [batch, seq])
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.squeeze(-1)
    
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=14,
            num_samples=1
        )
    
    # Save predictions
    predictions = output.squeeze().cpu().numpy()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save predictions and input
    np.save(output_path / "predictions.npy", predictions)
    np.save(output_path / "input.npy", input_data)
    
    print(f"\nSaved:")
    print(f"  input.npy: shape={input_data.shape}")
    print(f"  predictions.npy: shape={predictions.shape}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'input_shape': list(input_data.shape),
        'output_shape': list(predictions.shape),
    }
    
    import json
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nReference generation complete!")


def generate_rope_reference(output_dir, dim=64, max_position=1000, base=10000.0):
    """
    Generate RoPE-specific reference tensors.
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
        # cos and sin should be broadcastable to x's shape
        # Shape: [batch, num_heads, seq_len, dim]
        cos = cos.unsqueeze(0).unsqueeze(0)[:,:,-x.shape[2]:,:]
        sin = sin.unsqueeze(0).unsqueeze(0)[:,:,-x.shape[2]:,:]
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
                        help='Generate full model predictions')
    
    args = parser.parse_args()
    
    if args.rope_only:
        generate_rope_reference(args.output)
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
            generate_full_model_reference(args.input, args.output, args.model)
        else:
            # Default: generate both RoPE and full model references
            rope_dir = os.path.join(args.output, "rope")
            os.makedirs(rope_dir, exist_ok=True)
            generate_rope_reference(rope_dir)
            
            if args.input:
                generate_full_model_reference(args.input, args.output, args.model)


if __name__ == '__main__':
    main()
