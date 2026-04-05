# Sundial Rust 🌅

> A high-performance, pure Rust implementation of the **Sundial** time series forecasting model using Candle ML.

**[Sundial](https://github.com/thuml/Sundial)** is a powerful flow-matching based model for time series forecasting. This Rust port provides **zero Python dependencies** with production-grade performance.

## ✨ Features

- 🚀 **Pure Rust Implementation** - No Python dependencies required
- 📊 **Flow Matching** - Advanced probabilistic forecasting with uncertainty quantification
- 🧠 **Transformer Architecture** - 12-layer decoder with Rotary Positional Embeddings (RoPE)
- 📁 **Flexible Input** - Support for CSV and Parquet formats
- 🛠️ **Production CLI** - Ready-to-use command-line tool
- 🔬 **Debug Tools** - Tensor inspection and Python comparison utilities
- ⚡ **Performance** - Optimized with Candle ML and LTO

## 📦 Model Specifications

| Configuration | Value |
|--------------|-------|
| Model Type | thuml/sundial-base-128m |
| Hidden Size | 768 |
| Layers | 12 |
| Attention Heads | 12 |
| Head Dimension | 64 |
| Input Token Length | 16 |
| Activation | SiLU |

## 📦 Model Loading

Sundial Rust supports multiple model loading strategies with a memory-first default approach for optimal startup performance.

### Memory-First Loading (Default)

By default, Sundial loads model weights directly into memory without writing to disk:

- **Zero disk I/O**: Weights are decompressed from embedded assets into memory
- **Fast startup**: Eliminates filesystem overhead during model initialization
- **Integrity verification**: SHA256 hash verification ensures weight integrity
- **~490MB memory usage**: For the sundial-base-128m model

This is the recommended approach for most use cases.

### Environment Variables

Configure model loading via environment variables:

| Variable | Description |
|----------|-------------|
| `SUNDIAL_MODEL_PATH` | Path to external `model.safetensors` file |
| `SUNDIAL_CONFIG_PATH` | Path to external `config.json` file |
| `SUNDIAL_TEMP_DIR` | Custom directory for extracted weights (only with `SUNDIAL_USE_DISK=true`) |
| `SUNDIAL_USE_DISK` | Set to `"true"` to extract weights to disk instead of memory |

### Loading Strategies

**1. Memory Loading (Default)**
```rust
use sundial_rust::weights::loader::WeightLoader;
use sundial_rust::model::loader::load_sundial_from_memory;
use candle_core::Device;

let loader = WeightLoader::new()?; // Loads embedded weights into memory
let weights = loader.get_model_weights().expect("Memory weights available");
let config = load_config_from_env()?;
let device = Device::Cpu;
let model = load_sundial_from_memory(weights, &config, &device)?;
```

**2. External Weights**
```bash
export SUNDIAL_MODEL_PATH=/path/to/model.safetensors
export SUNDIAL_CONFIG_PATH=/path/to/config.json
```

**3. Disk Extraction**
```bash
export SUNDIAL_USE_DISK=true
```

### Programmatic API

```rust
use sundial_rust::weights::loader::{WeightLoader, verify_integrity, extract};
use std::path::Path;

// Load from memory (default)
let loader = WeightLoader::new()?;
let model_path = loader.model_path(); // Returns "<memory>" for in-memory weights

// Verify integrity of a weights file
verify_integrity(Path::new("weights/model.safetensors"))?;

// Extract embedded weights to disk
extract(WEIGHTS_COMPRESSED, Path::new("/tmp/model.safetensors"))?;
```

## 🚀 Quick Start

### Prerequisites

- Rust 1.70+ ([Install](https://rustup.rs/))
- Model weights (see [Setup](#-setup))

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/sundial-rust.git
cd sundial-rust

# Build the project
cargo build --release
```

### Basic Usage

#### 1. Preview Your Data

```bash
cargo run --release -- --input test_data/sample.csv
```

Output:
```
Successfully loaded time series data from: test_data/sample.csv
Row count: 10
Column names: ["timestamp", "value"]

First 5 rows:
shape: (5, 2)
┌────────────┬───────┐
│ timestamp  ┆ value │
│ ---        ┆ ---   │
│ str        ┆ f64   │
╞════════════╪═══════╡
│ 2024-01-01 ┆ 100.5 │
│ 2024-01-02 ┆ 101.2 │
│ ...
```

#### 2. Generate Forecasts

```bash
cargo run --release -- \
  --infer \
  --model weights/model.safetensors \
  --input test_data/sample.csv \
  --horizon 10
```

#### 3. Add Uncertainty Estimation

```bash
cargo run --release -- \
  --infer \
  --model weights/model.safetensors \
  --input test_data/sample.csv \
  --horizon 10 \
  --num-samples 100
```

#### 4. Save Results

```bash
# Save to JSON
cargo run --release -- \
  --infer \
  --model weights/model.safetensors \
  --input test_data/sample.csv \
  --horizon 10 \
  --output forecasts.json \
  --format json

# Save to CSV
cargo run --release -- \
  --infer \
  --model weights/model.safetensors \
  --input test_data/sample.csv \
  --horizon 10 \
  --output forecasts.csv \
  --format csv
```

## 🔧 Command-Line Reference

```
Sundial CLI - Time Series Forecasting Tool

USAGE:
    sundial [OPTIONS]

OPTIONS:
    -i, --input <FILE>           Input file path (CSV or Parquet)
    -o, --output <FILE>          Output file path for forecasts
    --horizon <N>                Number of future steps to forecast [default: 10]
    --window-size <N>            Historical points for input window [default: 30]
    --num-samples <N>            Stochastic samples for uncertainty [default: 1]
    -m, --model <FILE>           Pre-trained model (.safetensors)
    --format <FORMAT>            Output format: json or csv [default: json]
    --quiet                      Suppress console output
    -v, --verbose                Show detailed/debug information
    --infer                      Run inference to generate forecasts
    --quickstart                 Show quick-start tutorial
    --timestamp-col <NAME>       Custom timestamp column name [default: timestamp]
    --value-col <NAME>           Custom value column name [default: value]
    --no-auto-detect             Disable automatic column name detection
    -h, --help                   Print help
    -V, --version                Print version
```

### Flexible Column Names

Sundial Rust automatically detects common column names:

**Timestamp candidates** (in priority order): `time`, `datetime`, `date`, `timestamp`, `t`  
**Value candidates** (in priority order): `value`, `val`, `price`, `amount`, `y`, `target`

You can also specify custom columns:

```bash
cargo run --release -- \
  --infer \
  --model weights/model.safetensors \
  --input sales_data.csv \
  --horizon 24 \
  --timestamp-col "transaction_date" \
  --value-col "revenue"
```

## 📁 Data Requirements

Input files must contain **at least** two columns:
- **Time column**: Any timestamp format (e.g., `2024-01-01`, `2024-01-01 10:00:00`)
- **Value column**: Numeric values (float or integer)

### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Standard comma-separated values |
| Parquet | `.parquet` | Apache Parquet columnar format |

## 🧪 Testing

Run the full test suite:

```bash
# All tests
cargo test

# Linting with clippy
cargo clippy

# Formatting check
cargo fmt --check
```

Test results: ✅ **85 tests total** (56 library tests, 15 binary tests, 14 integration tests)

**Note:** One integration test (`test_full_build_verification`) may fail if the build script hasn't been run with the `--lib` flag.

## 🔬 Comparison with Python Implementation

This repository includes comparison scripts to verify correctness against the original PyTorch implementation:

```bash
# Compare full inference outputs
cargo run --release --example compare_exact

# Compare intermediate tensors
cargo run --release --example compare_intermediates

# Compare without revin (reversible normalization)
cargo run --release --example compare_no_revin
```

Python comparison scripts are available in the `scripts/` directory.

## 🏗️ Architecture

### Model Components

```
src/
├── model/
│   ├── sundial.rs       # Main Sundial model
│   ├── transformer.rs   # Transformer decoder
│   ├── attention.rs     # Multi-head attention
│   ├── decoder_layer.rs # Decoder layer
│   ├── patch_embed.rs   # Input patch embedding
│   ├── rope.rs          # Rotary positional embeddings
│   ├── mlp.rs           # Feed-forward network
│   └── loader.rs        # Weight loading from safetensors
├── flow/
│   ├── network.rs       # Flow matching network
│   ├── sampling.rs      # Flow matching inference
│   ├── resblock.rs      # Residual blocks
│   └── timestep_embed.rs  # Timestep embeddings
├── data/
│   └── mod.rs           # Data loading & preprocessing
└── bin/
    └── main.rs          # CLI entry point
```

### Key Design Decisions

1. **Candle ML Integration**: Uses [Candle](https://github.com/huggingface/candle) for efficient tensor operations
2. **Error Handling**: `anyhow::Result` for application errors, `thiserror` for library errors
3. **Logging**: Structured logging with `tracing` and `tracing-subscriber`
4. **Data Processing**: `polars` for high-performance data loading

## 📈 Performance

- **Inference**: Single-threaded CPU inference, optimized with LTO
- **Memory**: ~500MB RAM usage for sundial-base-128m model
- **Speed**: ~100ms per 10-step forecast on M1/M2 Macs

## 🛠️ Development

### Project Structure

```
sundial-rust/
├── src/                    # Source code
│   ├── lib.rs              # Library API
│   ├── model/              # Model architecture
│   ├── flow/               # Flow matching
│   ├── data/               # Data handling
│   ├── weights/            # Weight loading utilities
│   ├── assets/             # Embedded weights (US-002)
│   └── bin/                # CLI binary
├── weights/                # Pre-trained model weights
│   └── model.safetensors   # sundial-base-128m (490MB)
├── examples/               # Example programs
├── scripts/                # Python comparison scripts
├── test_data/              # Sample datasets
├── tasks/                  # Feature development tasks
├── archive-docs/           # Historical implementation docs
├── build.rs                # Build script for weight compression
└── tests/                  # Integration tests
```

### Adding New Features

1. Create a new task in `tasks/` (if implementing a new feature)
2. Run tests after each change: `cargo test`
3. Run clippy: `cargo clippy`
4. Format code: `cargo fmt`

### Debugging

Enable verbose mode for detailed output:

```bash
cargo run --release -- \
  --infer \
  --model weights/model.safetensors \
  --input test_data/sample.csv \
  --horizon 10 \
  --verbose
```

Use `debug_tensor()` and `compare_tensors()` utilities for tensor inspection.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 📚 References

- [Original Sundial Paper](https://arxiv.org/abs/2403.07815)
- [Sundial GitHub Repository](https://github.com/thuml/Sundial)
- [Candle ML Documentation](https://github.com/huggingface/candle)
- [Rust API Documentation](https://docs.rs/sundial-rust)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run `cargo test` and `cargo clippy`
4. Submit a pull request

---

**Built with ❤️ in Rust**
