# Sundial Rust Implementation Guide

## Architecture Overview

Sundial is a generative time series foundation model with three main components:

1. **Patch Embedding Layer** - Converts raw time series into patch tokens
2. **Causal Transformer** - Decoder-only transformer with RoPE for token representations
3. **Flow Matching Head** - Generates multiple probabilistic forecasts using flow-matching

## Model Configuration (from HuggingFace)

```rust
pub struct SundialConfig {
    pub input_token_len: usize,        // 16 (patch length)
    pub hidden_size: usize,            // 768
    pub intermediate_size: usize,      // 3072
    pub num_hidden_layers: usize,      // 12
    pub num_attention_heads: usize,    // 12
    pub hidden_act: String,            // "silu"
    pub dropout_rate: f64,             // 0.1
    pub max_position_embeddings: usize,// 10000
    pub output_token_lens: Vec<usize>, // [720]
    pub flow_loss_depth: usize,        // 3
    pub num_sampling_steps: usize,     // 50
    pub diffusion_batch_mul: usize,    // 4
}
```

## Implementation Plan

### Phase 1: Core Components

#### 1.1 Patch Embedding (`model/patch_embed.rs`)
```rust
// Key operations:
// 1. Pad input to be divisible by input_token_len
// 2. Unfold into patches of size input_token_len
// 3. Concatenate patch values with mask
// 4. Linear projection: (patch_len * 2) -> intermediate_size -> hidden_size
// 5. Add residual connection from linear projection of (patch_len * 2) -> hidden_size
```

#### 1.2 Rotary Positional Embeddings (`model/rope.rs`)
```rust
// Standard RoPE implementation for causal attention
// Pre-compute cos/sin caches up to max_position_embeddings
```

#### 1.3 Attention Layer (`model/attention.rs`)
```rust
// Multi-head attention with:
// - Q, K, V projections (bias=True)
// - Output projection (bias=False)
// - RoPE application
// - Causal attention mask
// - Scaled dot-product attention
```

#### 1.4 MLP Layer (`model/mlp.rs`)
```rust
// SwiGLU-style MLP:
// gate_proj: hidden_size -> intermediate_size (bias=False)
// up_proj: hidden_size -> intermediate_size (bias=False)  
// down_proj: intermediate_size -> hidden_size (bias=False)
// Output: down_proj(silu(gate_proj(x)) * up_proj(x))
```

#### 1.5 Decoder Layer (`model/decoder_layer.rs`)
```rust
// Standard decoder layer with:
// - Pre-normalization (LayerNorm)
// - Self-attention with residual
// - FFN with residual
```

#### 1.6 Transformer Backbone (`model/transformer.rs`)
```rust
// Stack of decoder layers
// Final LayerNorm
// Support for KV cache for efficient generation
```

### Phase 2: Flow Matching Head

#### 2.1 Timestep Embedding (`flow/timestep_embed.rs`)
```rust
// Sinusoidal timestep embedding
// MLP: freq_dim -> hidden_size -> hidden_size with SiLU
```

#### 2.2 ResBlock with AdaLN (`flow/resblock.rs`)
```rust
// Residual block with adaptive layer normalization
// Modulation from timestep + condition embedding
```

#### 2.3 Flow Network (`flow/network.rs`)
```rust
// SimpleMLPAdaLN:
// - Input projection: in_channels -> model_channels
// - Timestep embedding
// - Condition embedding: z_channels -> model_channels
// - Residual blocks with AdaLN modulation
// - Final layer with modulation
```

#### 2.4 Flow Sampling (`flow/sampling.rs`)
```rust
// Euler integration for flow matching:
// 1. Start with noise ~ N(0, I)
// 2. For each sampling step:
//    - Compute velocity prediction
//    - Update: x = x + (pred - noise) * dt
// 3. Reshape to [num_samples, forecast_length]
```

### Phase 3: Model Integration

#### 3.1 Main Model (`model/sundial.rs`)
```rust
pub struct SundialModel {
    config: SundialConfig,
    embed_layer: SundialPatchEmbedding,
    layers: Vec<SundialDecoderLayer>,
    norm: LayerNorm,
    flow_loss: FlowLoss,
}

impl SundialModel {
    // Forward pass for encoding
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor>
    
    // Generate forecasts
    pub fn generate(
        &self, 
        input_ids: &Tensor, 
        max_new_tokens: usize,
        num_samples: usize,
        revin: bool
    ) -> Result<Tensor>
}
```

### Phase 4: Data Loading & CLI

#### 4.1 Data Fetching (`data/yfinance.rs`)
```rust
// Use reqwest to call yfinance API or similar
// Download historical price data
```

#### 4.2 Preprocessing (`data/preprocess.rs`)
```rust
// ReNorm (RevIN): normalize using mean/std of input
// Create patch sequences
```

#### 4.3 CLI Interface (`main.rs`)
```rust
// Command line interface for forecasting
// Support multiple tickers, forecast lengths, output formats
```

## Key Differences from Python Implementation

1. **Weight Loading**: Use `candle`'s safetensors support instead of PyTorch pickle
2. **Memory Management**: Explicit tensor lifetimes, no garbage collector
3. **Error Handling**: Use `Result<T, E>` instead of exceptions
4. **Parallelism**: Use Rayon or native async for batch operations
5. **No Dynamic Graph**: All operations are static, better optimization

## Testing Strategy

1. **Unit Tests**: Test each layer against known outputs
2. **Integration Tests**: Compare full model output with Python reference
3. **Numerical Precision**: Ensure FP32 matches PyTorch FP32
4. **Performance Benchmarks**: Compare inference speed

## File Structure

```
sundial-rust/
├── Cargo.toml
├── src/
│   ├── main.rs              # CLI entry point
│   ├── lib.rs               # Library exports
│   ├── model/
│   │   ├── mod.rs
│   │   ├── config.rs        # SundialConfig
│   │   ├── patch_embed.rs   # Patch embedding layer
│   │   ├── rope.rs          # Rotary embeddings
│   │   ├── attention.rs     # Self-attention
│   │   ├── mlp.rs           # SwiGLU MLP
│   │   ├── decoder_layer.rs # Decoder layer
│   │   ├── transformer.rs   # Transformer backbone
│   │   └── sundial.rs       # Main model
│   ├── flow/
│   │   ├── mod.rs
│   │   ├── timestep_embed.rs
│   │   ├── resblock.rs
│   │   ├── network.rs       # Flow network
│   │   └── sampling.rs      # Flow sampling
│   └── data/
│       ├── mod.rs
│       ├── loader.rs        # Data loading
│       └── preprocess.rs    # RevIN normalization
├── tests/
│   └── integration_test.rs
└── examples/
    └── spy_forecast.rs
```

## Implementation Order

1. ✅ Config structure (already done)
2. ✅ Basic data structures (TimeSeriesData)
3. ⏳ Patch embedding layer
4. ⏳ Rotary embeddings
5. ⏳ Attention mechanism
6. ⏳ MLP layer
7. ⏳ Decoder layer
8. ⏳ Transformer backbone
9. ⏳ Flow network components
10. ⏳ Flow sampling
11. ⏳ Main model integration
12. ⏳ Weight loading from safetensors
13. ⏳ Data fetching
14. ⏳ CLI interface
15. ⏳ Testing & validation

## Dependencies to Add

```toml
[dependencies]
# Already have: candle-core, candle-nn, candle-transformers

# Add:
candle-flash-attn = "0.1"  # Optional: Flash attention for speed
rayon = "1.8"              # Parallel processing
indicatif = "0.17"         # Progress bars
csv = "1.3"                # CSV output
```

## Notes

- The model uses **patch-based** processing: input is split into 16-timestep patches
- **RevIN** (Reversible Instance Normalization) is applied before encoding and reversed after decoding
- **Flow matching** generates probabilistic forecasts by solving an ODE from noise to data
- The model supports **variable-length** forecasts through multi-patch prediction
- **KV caching** enables efficient autoregressive generation
