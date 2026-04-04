#!/usr/bin/env python3
"""
Python script to capture intermediate tensors from Sundial model.
Used for comparing with Rust implementation.

Usage:
    python scripts/compare_intermediates.py --output /tmp/python_intermediates.npz
"""

import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def make_hook(name, intermediates):
    """Create a hook to capture intermediate outputs."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            # Usually output[0] is the hidden states
            intermediates[name] = output[0].detach().cpu().numpy()
        else:
            intermediates[name] = output.detach().cpu().numpy()
    return hook


def main():
    parser = argparse.ArgumentParser(description="Capture intermediate tensors from Sundial")
    parser.add_argument("--model", type=str, default="thuml/sundial-base-128m",
                        help="Model name or path")
    parser.add_argument("--output", type=str, default="/tmp/python_intermediates.npz",
                        help="Output file path")
    parser.add_argument("--seq-len", type=int, default=2880,
                        help="Input sequence length")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Stop after this many layers (for debugging)")
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    model.eval()
    
    # Dictionary to store intermediates
    intermediates = {}
    
    # Hook the patch embedding
    if hasattr(model.model, 'embed_layer'):
        model.model.embed_layer.register_forward_hook(
            make_hook('patch_embed_output', intermediates)
        )
    
    # Hook transformer layers
    layer_count = 0
    if hasattr(model.model, 'layers'):
        for i, layer in enumerate(model.model.layers):
            if args.num_layers is not None and i >= args.num_layers:
                break
            
            # Register hooks for the layer
            layer.register_forward_hook(make_hook(f'layer_{i}_output', intermediates))
            
            # Hook attention sub-components if available
            if hasattr(layer, 'self_attn'):
                layer.self_attn.register_forward_hook(
                    make_hook(f'layer_{i}_attention', intermediates)
                )
            
            layer_count += 1
    
    # Create random input (matching Rust test)
    input_ids = torch.randn(args.batch_size, args.seq_len)
    
    print(f"Running inference with input shape: {input_ids.shape}")
    
    # Run inference
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_new_tokens=14, 
            num_samples=1, 
            revin=False
        )
    
    # Save intermediates
    print(f"Saving {len(intermediates)} intermediate tensors to {args.output}")
    np.savez(args.output, **intermediates)
    
    # Print summary
    print("\nCaptured tensors:")
    for name, tensor in sorted(intermediates.items()):
        print(f"  {name}: shape={tensor.shape}, mean={tensor.mean():.6f}, std={tensor.std():.6f}")
    
    print(f"\nSaved to: {args.output}")
    print(f"Total layers processed: {layer_count}")


if __name__ == "__main__":
    main()
