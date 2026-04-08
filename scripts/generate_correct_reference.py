#!/usr/bin/env python3
"""
Generate CORRECT Python reference tensors using actual model weights.

This script manually loads weights from the safetensors file to ensure
the reference data is generated with the correct weights.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from pathlib import Path
import sys
sys.path.insert(0, '/Users/ongjingrong/Documents/Projects/sundial/sundial-rust')

from sundial_model.modeling_sundial import SundialModel
from sundial_model.configuration_sundial import SundialConfig


def load_all_weights(model, safetensors_path="./weights/model.safetensors"):
    """Load all weights from safetensors into the model"""
    weights = load_file(safetensors_path)
    
    # Load embed_layer weights
    if 'model.embed_layer.hidden_layer.weight' in weights:
        model.embed_layer.hidden_layer.weight.data = weights['model.embed_layer.hidden_layer.weight']
        model.embed_layer.hidden_layer.bias.data = weights['model.embed_layer.hidden_layer.bias']
        model.embed_layer.output_layer.weight.data = weights['model.embed_layer.output_layer.weight']
        model.embed_layer.residual_layer.weight.data = weights['model.embed_layer.residual_layer.weight']
        model.embed_layer.residual_layer.bias.data = weights['model.embed_layer.residual_layer.bias']
    
    # Load decoder layer weights
    for layer_idx in range(model.config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        
        # Self-attention
        if f'{prefix}.self_attn.q_proj.weight' in weights:
            model.layers[layer_idx].self_attn.q_proj.weight.data = weights[f'{prefix}.self_attn.q_proj.weight']
            model.layers[layer_idx].self_attn.q_proj.bias.data = weights[f'{prefix}.self_attn.q_proj.bias']
            model.layers[layer_idx].self_attn.k_proj.weight.data = weights[f'{prefix}.self_attn.k_proj.weight']
            model.layers[layer_idx].self_attn.k_proj.bias.data = weights[f'{prefix}.self_attn.k_proj.bias']
            model.layers[layer_idx].self_attn.v_proj.weight.data = weights[f'{prefix}.self_attn.v_proj.weight']
            model.layers[layer_idx].self_attn.v_proj.bias.data = weights[f'{prefix}.self_attn.v_proj.bias']
            model.layers[layer_idx].self_attn.o_proj.weight.data = weights[f'{prefix}.self_attn.o_proj.weight']
        
        # MLP
        if f'{prefix}.ffn_layer.gate_proj.weight' in weights:
            model.layers[layer_idx].ffn_layer.gate_proj.weight.data = weights[f'{prefix}.ffn_layer.gate_proj.weight']
            model.layers[layer_idx].ffn_layer.up_proj.weight.data = weights[f'{prefix}.ffn_layer.up_proj.weight']
            model.layers[layer_idx].ffn_layer.down_proj.weight.data = weights[f'{prefix}.ffn_layer.down_proj.weight']
        
        # Layer norms
        if f'{prefix}.norm1.weight' in weights:
            model.layers[layer_idx].norm1.weight.data = weights[f'{prefix}.norm1.weight']
            model.layers[layer_idx].norm1.bias.data = weights[f'{prefix}.norm1.bias']
            model.layers[layer_idx].norm2.weight.data = weights[f'{prefix}.norm2.weight']
            model.layers[layer_idx].norm2.bias.data = weights[f'{prefix}.norm2.bias']
    
    # Load final norm
    if 'model.norm.weight' in weights:
        model.norm.weight.data = weights['model.norm.weight']
        model.norm.bias.data = weights['model.norm.bias']
    
    print(f"Loaded all weights from {safetensors_path}")


def make_hook(storage_dict, name):
    """Create a forward hook to capture intermediate outputs"""
    def hook(module, input, output):
        if isinstance(output, tuple):
            storage_dict[name] = output[0].detach().cpu().numpy()
        else:
            storage_dict[name] = output.detach().cpu().numpy()
    return hook


def main():
    print("=== Generating CORRECT Python Reference Tensors ===\n")
    
    # Load input
    input_path = 'tests/reference_data/intermediates/input.npy'
    input_data = np.load(input_path)
    input_tensor = torch.from_numpy(input_data).float()
    print(f"Loaded input: shape={input_tensor.shape}")
    
    # Create config and model
    config = SundialConfig()
    model = SundialModel(config)
    model.eval()
    
    # Load actual weights
    load_all_weights(model)
    
    # Create storage for intermediates
    intermediates = {}
    
    # Register hooks
    print("\nRegistering hooks...")
    model.embed_layer.register_forward_hook(make_hook(intermediates, 'patch_embed_output'))
    
    for i in range(len(model.layers)):
        model.layers[i].register_forward_hook(make_hook(intermediates, f'layer_{i}_output'))
    
    if hasattr(model, 'norm'):
        model.norm.register_forward_hook(make_hook(intermediates, 'transformer_output'))
    
    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output = model(input_tensor)
    
    # Save predictions
    if hasattr(output, 'last_hidden_state'):
        intermediates['predictions'] = output.last_hidden_state.detach().cpu().numpy()
    elif isinstance(output, tuple):
        intermediates['predictions'] = output[0].detach().cpu().numpy()
    else:
        intermediates['predictions'] = output.detach().cpu().numpy()
    
    # Also save the input for reference
    intermediates['input'] = input_tensor.cpu().numpy()
    
    # Create output directory
    output_dir = Path('tests/reference_data/intermediates')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all intermediates
    print(f"\nSaving {len(intermediates)} tensors to: {output_dir}")
    for name, tensor in intermediates.items():
        save_path = output_dir / f"{name}.npy"
        np.save(save_path, tensor)
        print(f"  Saved {name:30s}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    # Save metadata
    metadata = {
        'model_name': 'sundial-base-128m',
        'input_shape': input_tensor.shape,
        'output_shape': output.last_hidden_state.shape if hasattr(output, 'last_hidden_state') else output.shape,
        'num_layers': len(model.layers),
        'tensor_names': list(intermediates.keys()),
    }
    
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nReference generation complete!")
    print(f"Total tensors saved: {len(intermediates)}")


if __name__ == '__main__':
    main()
