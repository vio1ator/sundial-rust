#!/usr/bin/env python3
"""
Debug patch embedding by comparing Python vs Rust step by step.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load input
input_data = np.load('tests/reference_data/intermediates/input.npy')
input_tensor = torch.from_numpy(input_data).float()

print(f"Input shape: {input_tensor.shape}")

# Patch embedding parameters from Sundial config
input_token_len = 16
hidden_size = 768
intermediate_size = 3072
dropout_rate = 0.0

# Create the patch embedding layers with random weights for now
# We'll use the actual weights later
torch.manual_seed(42)

input_dim = input_token_len * 2  # values + mask

# Create layers
hidden_layer = nn.Linear(input_dim, intermediate_size)
act_fn = F.silu  # SiLU
output_layer = nn.Linear(intermediate_size, hidden_size, bias=False)
residual_layer = nn.Linear(input_dim, hidden_size)

print(f"\n=== Forward Pass ===")
print(f"Input shape: {input_tensor.shape}")

# Step 1: Pad input
x = input_tensor
batch_size = x.shape[0]
input_length = x.shape[-1]
padding_length = (input_token_len - (input_length % input_token_len)) % input_token_len
print(f"Padding length: {padding_length}")

x_padded = F.pad(x, (padding_length, 0))
print(f"Padded shape: {x_padded.shape}")

# Step 2: Create mask
mask = torch.ones_like(x_padded, dtype=torch.float32)
print(f"Mask shape: {mask.shape}")

# Step 3: Unfold
x_patches = x_padded.unfold(dimension=-1, size=input_token_len, step=input_token_len)
mask_patches = mask.unfold(dimension=-1, size=input_token_len, step=input_token_len)

print(f"X patches shape: {x_patches.shape}")
print(f"Mask patches shape: {mask_patches.shape}")

# Step 4: Reshape patches before concatenation
batch, num_patches, patch_len = x_patches.shape
print(f"batch={batch}, num_patches={num_patches}, patch_len={patch_len}")

x_patches_reshaped = x_patches.reshape(batch * num_patches, patch_len)
mask_patches_reshaped = mask_patches.reshape(batch * num_patches, patch_len)

print(f"x_patches_reshaped: {x_patches_reshaped.shape}")
print(f"mask_patches_reshaped: {mask_patches_reshaped.shape}")

# Step 5: Concatenate
x_concat = torch.cat([x_patches_reshaped, mask_patches_reshaped], dim=1)
print(f"Concatenated shape: {x_concat.shape}")

# Save unfolded x for comparison
np.save('/tmp/x_patches_python.npy', x_patches.detach().cpu().numpy())
np.save('/tmp/mask_patches_python.npy', mask_patches.detach().cpu().numpy())
np.save('/tmp/x_patches_reshaped_python.npy', x_patches_reshaped.detach().cpu().numpy())
np.save('/tmp/mask_patches_reshaped_python.npy', mask_patches_reshaped.detach().cpu().numpy())
np.save('/tmp/x_concat_python.npy', x_concat.detach().cpu().numpy())

print(f"\nSaved intermediate tensors to /tmp/")

# Step 7: Apply hidden layer + activation
hid = hidden_layer(x_concat)
print(f"After hidden layer: {hid.shape}")

hid = act_fn(hid)
print(f"After activation: {hid.shape}")

# Save hidden
np.save('/tmp/hid_python.npy', hid.detach().cpu().numpy())

# Step 8: Apply output layer
out = output_layer(hid)
print(f"After output layer: {out.shape}")

np.save('/tmp/out_python.npy', out.detach().cpu().numpy())

# Step 9: Apply residual
res = residual_layer(x_concat)
print(f"Residual shape: {res.shape}")

out = out + res
print(f"After residual: {out.shape}")

np.save('/tmp/out_residual_python.npy', out.detach().cpu().numpy())

# Step 10: Reshape back
out = out.reshape(batch, num_patches, hidden_size)
print(f"Final output shape: {out.shape}")

# Save final output
np.save('/tmp/patch_embed_output_debug.npy', out.detach().cpu().numpy())

print(f"\n=== All intermediates saved ===")
print("Files saved to /tmp/:")
print("  - x_patches_python.npy")
print("  - mask_patches_python.npy")
print("  - x_concat_python.npy")
print("  - x_reshaped_python.npy")
print("  - hid_python.npy")
print("  - out_python.npy")
print("  - out_residual_python.npy")
print("  - patch_embed_output_debug.npy")
