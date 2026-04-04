# Loading Pretrained Sundial Weights

This guide explains how to download and load the pretrained Sundial model weights from HuggingFace.

## Prerequisites

1. Install `huggingface-cli`:
```bash
pip install huggingface_hub
```

2. Alternatively, download manually using curl/wget.

## Download Model Files

### Option 1: Using huggingface-cli (Recommended)

```bash
huggingface-cli download thuml/sundial-base-128m --local-dir ./model_cache
```

This will download:
- `model.safetensors` (513 MB)
- `config.json`
- `configuration_sundial.py`
- Other metadata files

### Option 2: Manual Download

```bash
mkdir -p model_cache
curl -L https://huggingface.co/thuml/sundial-base-128m/resolve/main/model.safetensors -o model_cache/model.safetensors
curl -L https://huggingface.co/thuml/sundial-base-128m/resolve/main/config.json -o model_cache/config.json
```

## Load Model in Rust

### Basic Usage

```rust
use candle_core::Device;
use sundial_rust::{SundialConfig, SundialModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    
    // Load model from safetensors
    let config = SundialConfig::default();
    let model = SundialModel::load_from_safetensors(
        config,
        "./model_cache/model.safetensors",
        &device,
    )?;
    
    // Use the model for inference
    // ...
    
    Ok(())
}
```

### Using the CLI Example

```bash
# Build the example
cargo build --release

# Run with pretrained weights
./target/release/sundial-forecast \
  --ticker SPY \
  --forecast-length 14 \
  --num-samples 100 \
  --model-path ./model_cache/model.safetensors
```

## Model Architecture

The Sundial model has the following configuration:

```json
{
  "input_token_len": 16,
  "hidden_size": 768,
  "intermediate_size": 3072,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "hidden_act": "silu",
  "max_position_embeddings": 10000,
  "output_token_lens": [720],
  "flow_loss_depth": 3,
  "num_sampling_steps": 50,
  "diffusion_batch_mul": 4
}
```

## Weight Loading Implementation

The `SundialModel::load_from_safetensors` method:

1. Reads the safetensors file
2. Deserializes the tensor metadata
3. Loads all tensors onto the specified device
4. Creates the model structure
5. Maps loaded tensors to model variables

**Note**: The current implementation loads the tensors but doesn't fully map them to the model variables. This requires implementing a proper weight mapping system.

## Next Steps: Complete Weight Mapping

To fully load the pretrained weights, you need to:

1. **Parse tensor names** from safetensors
2. **Map to model variables** in the Sundial architecture
3. **Assign weights** to the corresponding layers

Example mapping structure:

```rust
// Tensor name patterns in safetensors:
// - model.embed_layer.hidden_layer.weight
// - model.embed_layer.output_layer.weight
// - model.layers.0.self_attn.q_proj.weight
// - model.layers.0.self_attn.k_proj.weight
// - model.layers.0.self_attn.v_proj.weight
// - model.layers.0.self_attn.o_proj.weight
// - model.layers.0.ffn_layer.gate_proj.weight
// - model.layers.0.ffn_layer.up_proj.weight
// - model.layers.0.ffn_layer.down_proj.weight
// - flow_loss.time_embed.mlp.0.weight
// - flow_loss.cond_embed.weight
// - etc.
```

## Troubleshooting

### Issue: "Tensor not found"

Make sure the safetensors file is complete and not corrupted. Re-download if necessary.

### Issue: "Shape mismatch"

The tensor shapes in the safetensors file should match the model architecture. Verify your configuration matches the pretrained model.

### Issue: "Device placement error"

Ensure all tensors are loaded on the same device (CPU or GPU).

## Performance Tips

1. **Use GPU**: Load tensors on GPU for faster inference
```rust
#[cfg(feature = "cuda")]
let device = Device::Cuda(0)?;
```

2. **Memory mapping**: For large models, consider memory-mapped loading
```rust
use candle_core::safetensors::MmapedSafetensors;
let safetensors = MmapedSafetensors::new(path)?;
```

3. **Quantization**: Consider using FP16 or INT8 for reduced memory usage

## References

- [HuggingFace Model Page](https://huggingface.co/thuml/sundial-base-128m)
- [Safetensors Documentation](https://github.com/huggingface/safetensors)
- [Candle Safetensors](https://docs.rs/candle-core/latest/candle_core/safetensors/)
