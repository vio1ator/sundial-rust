---
datasets:
- thuml/UTSD
- Salesforce/lotsa_data
- autogluon/chronos_datasets
license: apache-2.0
metrics:
- mse
- mae
- mase
- wql
- crps
pipeline_tag: time-series-forecasting
tags:
- time series
- time-series
- forecasting
- foundation models
- pretrained models
- generative models
- time series foundation models
library_name: transformers
---

# Sundial (Timer 3.0) - Rust Implementation

🚩 **News (2025.08)** Sundial has been integrated into [Apache IoTDB](https://iotdb.apache.org/), a native time-series database.

🚩 **News (2025.06)** Sundial has been accepted as **ICML 2025 Oral** (Top 1%).

🚩 **News (2025.05)** Get **1st MASE** on the [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) Benchmark.

🚩 **News (2025.02)** Get **1st MSE/MAE** zero-shot performance on [Time-Series-Library](https://github.com/thuml/Time-Series-Library) datasets.

![Sundial Architecture](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/xoSJYO6GSHeFKY9eLjNz2.png)

Sundial is a family of **generative** time series foundation models. This version is pre-trained on **1 trillion** time points with **128M** parameters. For more information, please refer to this [paper](https://arxiv.org/pdf/2502.00816).

The Rust implementation provides high-performance inference using Candle ML framework with:
- Zero-copy weight loading from embedded assets
- Flow matching for probabilistic forecasting
- RevIN (Reversible Instance Normalization)
- FlashAttention optimization
- KV Cache support

## Quickstart (Rust)

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
sundial-rust = "0.1.0"
candle-core = "0.8"
candle-nn = "0.8"
```

### Basic Usage

```rust
use sundial_rust::{SundialModel, SundialConfig};
use candle_core::{Tensor, Device};

// Load model with embedded weights (no disk I/O)
let config = SundialConfig::sundial_base_128m();
let device = Device::Cpu;
let model = SundialModel::load_from_safetensors(
    config,
    "path/to/weights",  // Falls back to embedded weights if path doesn't exist
    &device
)?;

// Prepare input: [batch_size, lookback_length]
// Standard lookback length is 2880 timesteps
let input = Tensor::randn(0.0, 1.0, (1, 2880), &device)?;

// Generate probabilistic forecasts
let predictions = model.generate(
    &input,
    720,        // forecast_length: number of future timesteps
    20,         // num_samples: number of probabilistic samples
    true,       // revin: apply normalization
)?;

// Output shape: [batch_size, num_samples, forecast_length]
println!("Predictions shape: {:?}", predictions.shape());
```

### Loading External Weights

```rust
// Set environment variable for external weights
std::env::set_var("SUNDIAL_MODEL_PATH", "/path/to/model.safetensors");

// Model will load from the specified path
let model = SundialModel::load_from_safetensors(config, "path/to/weights", &device)?;
```

### Loading Embedded Weights (Default)

```rust
// If no external path is set, embedded weights are loaded from memory
// This is the default behavior and requires no disk I/O
let model = SundialModel::load_from_safetensors(
    SundialConfig::sundial_base_128m(),
    "model.safetensors",  // This path is ignored for embedded mode
    &device
)?;
```

## Model Architecture

### Overview

Sundial can be viewed as an **ARMA** model (Auto-Regression and Moving-Average). Transformer learns auto-regressive token representations. Conditioned on them, TimeFlow transforms random noises into non-deterministic predictions.

![Flow Matching](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/B5w-TNPnTBpChexIhsVOp.png)

**Overall Architecture:**
1. **Patch Embeddings**: Input time series divided into patches (16 timesteps each)
2. **Decoder-only Transformer**: Causal attention with FlashAttention for efficiency
3. **TimeFlow Loss**: Parameterized loss modeling per-token probability distributions
4. **Flow Matching**: Generates multiple plausible predictions via Euler integration

### Specification

- **Architecture**: Causal Transformer (Decoder-only)
- **Pre-training Scale**: 1032B time points
- **Context Length**: up to 2880 timesteps
- **ReNorm**: Default=True (RevIN)
- **Patch Length**: 16 timesteps
- **Multi-Patch Prediction Length**: 720 timesteps
- **Parameter Count**: 128M
- **Number of Layers**: 12
- **Precision**: FP32
- **Speedup**: KV Cache & FlashAttention

### Flow Matching Sampling

The Rust implementation uses Euler integration for flow matching:

```
x_t+1 = x_t + (velocity - noise) * dt
```

Where:
- `velocity` is predicted by the flow network (SimpleMLPAdaLN)
- `noise` is the initial random sample ~ N(0, I)
- `dt = 1 / num_sampling_steps` (default: 50 steps)

This generates diverse probabilistic forecasts from the same input condition.

## Evaluation

We evaluate performance on the following benchmarks:

- [GIFT-Eval (1st MASE)](https://huggingface.co/spaces/Salesforce/GIFT-Eval)
- [Time-Series-Library (1st MSE/MAE)](https://github.com/thuml/Time-Series-Library)
- [FEV Leaderboard](https://huggingface.co/spaces/Salesforce/FEV)

## Inference Time (Rust Implementation)

### Apple M1 Pro CPU (16 GB)

| Lookback Length | Forecast Length | # Samples | Inference Time | Optimization |
| --------------- | --------------- | --------- | -------------- | ------------ |
| 672             | 16              | 1         | ~250ms         | -            |
| 2880            | 16              | 1         | ~510ms         | FlashAttention |
| 2880            | 720             | 1         | ~510ms         | Multi-Patch  |
| 2880            | 1440            | 1         | ~789ms         | KV Cache     |
| 2880            | 720             | 20        | ~950ms         | Shared Condition |

## Configuration

The `SundialConfig` struct provides full control over model parameters:

```rust
use sundial_rust::SundialConfig;

let config = SundialConfig {
    input_token_len: 16,           // Patch length
    hidden_size: 768,              // Embedding dimension
    intermediate_size: 3072,       // MLP dimension
    num_hidden_layers: 12,         // Transformer layers
    num_attention_heads: 12,       // Attention heads
    output_token_lens: vec![720],  // Forecast lengths supported
    hidden_act: "silu".to_string(), // Activation function
    use_cache: true,               // Enable KV cache
    rope_theta: 10000.0,           // RoPE base frequency
    dropout_rate: 0.1,             // Dropout rate
    initializer_range: 0.02,       // Weight initialization
    max_position_embeddings: 10000, // Max sequence length
    flow_loss_depth: 3,            // Flow network depth
    num_sampling_steps: 50,        // Flow sampling steps
    diffusion_batch_mul: 4,        // Diffusion batch multiplier
    debug_mode: false,             // Enable debug outputs
    debug_layer: None,             // Stop at specific layer for debugging
};
```

## Debugging

For debugging model discrepancies:

```rust
// Enable debug mode
let mut config = SundialConfig::sundial_base_128m();
config.debug_mode = true;
config.debug_layer = Some(6); // Stop after layer 6

let model = SundialModel::load_from_safetensors(config, "path/to/weights", &device)?;

// Use debug utilities to inspect tensors
use sundial_rust::debug_tensor;
debug_tensor(&intermediate, "layer_6_output");
```

## Citation

If you find Sundial helpful for your research, please cite our paper:

```bibtex
@article{liu2025sundial,
  title={Sundial: A Family of Highly Capable Time Series Foundation Models},
  author={Liu, Yong and Qin, Guo and Shi, Zhiyuan and Chen, Zhi and Yang, Caiyin and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2502.00816},
  year={2025}
}
```

## Contact

If you have any questions or want to use the code, feel free to contact:

- Yong Liu (liuyong21@mails.tsinghua.edu.cn)
- Guo Qin (qinguo24@mails.tsinghua.edu.cn)

## License

This model is licensed under the Apache-2.0 License.
