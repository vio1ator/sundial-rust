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

Test results: ✅ **50 tests passed**
- 37 unit tests (model components, data processing)
- 13 integration tests (CLI functionality, file I/O)

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
│   └── resblock.rs      # Residual blocks
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
│   └── bin/                # CLI binary
├── weights/                # Pre-trained model weights
│   └── model.safetensors   # sundial-base-128m (490MB)
├── examples/               # Example programs
├── scripts/                # Python comparison scripts
├── test_data/              # Sample datasets
└── tasks/                  # Feature development tasks
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
