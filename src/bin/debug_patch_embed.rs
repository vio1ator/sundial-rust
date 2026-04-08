//! Debug patch embedding layer

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::fs::File;
use std::io::Read;

fn main() -> Result<()> {
    println!("=== Debugging Patch Embedding Layer ===\n");

    // Load input
    let input = load_npy("tests/reference_data/intermediates/input.npy")?;
    println!("Input shape: {:?}", input.dims());

    // Load Python reference output
    let python_output = load_npy("tests/reference_data/intermediates/patch_embed_output.npy")?;
    println!("Python output shape: {:?}", python_output.dims());

    // Create patch embedding with same config
    let config = sundial_rust::SundialConfig::default();
    println!(
        "Config: input_token_len={}, hidden_size={}, intermediate_size={}",
        config.input_token_len, config.hidden_size, config.intermediate_size
    );

    // Load model weights
    let device = Device::Cpu;
    let vb = sundial_rust::model::loader::load_sundial_from_huggingface(
        "thuml/sundial-base-128m",
        &device,
    )?;

    // Extract just the patch embedding layer
    // We need to access it from the model
    // For now, let's manually test the unfold operation

    // Test unfold
    println!("\n=== Testing Unfold Operation ===");
    let x = &input;
    let batch_size = x.dim(0)?;
    let seq_len = x.dim(1)?;
    let patch_len = 16;
    let step = 16;
    let num_patches = seq_len / patch_len;

    println!(
        "seq_len={}, patch_len={}, step={}, num_patches={}",
        seq_len, patch_len, step, num_patches
    );

    // Manual unfold
    let mut patches = Vec::with_capacity(batch_size * num_patches * patch_len);
    for b in 0..batch_size {
        for p in 0..num_patches {
            let start = p * step;
            for i in 0..patch_len {
                let val = x.get(b)?.get(start + i)?.to_scalar::<f32>()?;
                patches.push(val);
            }
        }
    }

    let rust_patches = Tensor::from_vec(patches, (batch_size, num_patches, patch_len), &device)?;
    println!("Rust patches shape: {:?}", rust_patches.dims());

    // Compare with Python
    // Python should have done the same unfold
    // Let's check if they match

    let python_patches = load_npy("tests/reference_data/intermediates/patch_embed_output.npy")?;

    // Actually, we need to compare the full forward pass
    // Let's create a minimal patch embedding test

    println!("\n=== Testing Full Forward Pass ===");

    // For now, just save the input for manual testing
    save_tensor_to_bin("/tmp/patch_embed_input.bin", &input)?;

    println!("Input saved to /tmp/patch_embed_input.bin");

    Ok(())
}

fn load_npy(path: &str) -> Result<Tensor> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Parse numpy header
    let header_len = u16::from_le_bytes([buffer[8], buffer[9]]) as usize;
    let header_start = 10;
    let header = std::str::from_utf8(&buffer[header_start..header_start + header_len])?;

    // Extract shape
    let shape = parse_shape_from_header(header)?;

    // Extract data (f32, little-endian)
    let data_start = header_start + header_len;
    let data = &buffer[data_start..];

    let data_vec: Vec<f32> = data
        .chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let device = Device::Cpu;
    Ok(Tensor::from_vec(data_vec, shape.as_slice(), &device)?)
}

fn parse_shape_from_header(header: &str) -> anyhow::Result<Vec<usize>> {
    let shape_idx = header
        .find("shape")
        .ok_or_else(|| anyhow::anyhow!("No shape in header"))?;
    let rest = &header[shape_idx..];

    let paren_start = rest
        .find('(')
        .ok_or_else(|| anyhow::anyhow!("No paren in shape"))?;
    let paren_end = rest[paren_start..]
        .find(')')
        .ok_or_else(|| anyhow::anyhow!("No closing paren"))?
        + paren_start;

    let shape_str = &rest[paren_start + 1..paren_end];

    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if shape.is_empty() {
        return Err(anyhow::anyhow!("Failed to parse shape: {}", shape_str));
    }

    Ok(shape)
}

fn save_tensor_to_bin(path: &str, tensor: &Tensor) -> Result<()> {
    use std::io::Write;

    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    let dims = tensor.dims();

    let mut file = File::create(path)?;

    // Write dimension count
    file.write_all(&(dims.len() as u32).to_le_bytes())?;

    // Write dimensions
    for &d in dims {
        file.write_all(&(d as u32).to_le_bytes())?;
    }

    // Write data
    for &val in &data {
        file.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}
