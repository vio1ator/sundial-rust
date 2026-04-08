//! Test patch embedding layer with actual weights

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};
use std::fs::File;
use std::io::{Read, Write};

fn main() -> Result<()> {
    println!("=== Testing Patch Embedding with Actual Weights ===\n");

    // Load input
    let input = load_npy("tests/reference_data/intermediates/input.npy")?;
    println!("Input shape: {:?}", input.dims());

    // Load Python reference output
    let python_output = load_npy("tests/reference_data/intermediates/patch_embed_output.npy")?;
    println!("Python output shape: {:?}", python_output.dims());

    // Load weights from safetensors
    let vb = sundial_rust::model::loader::load_sundial_from_huggingface(
        "thuml/sundial-base-128m",
        &Device::Cpu,
    )?;

    // Get the patch embed weights manually
    // We need to extract them from the varb
    // For now, let's just run the full model and see what happens

    // Create patch embedding layer manually
    let config = sundial_rust::SundialConfig::default();
    let input_dim = config.input_token_len * 2; // 32

    // Extract weights from vb
    let hidden_weight = vb.get(
        (config.intermediate_size, input_dim),
        "embed_layer.hidden_layer.weight",
    )?;
    let hidden_bias = vb.get(config.intermediate_size, "embed_layer.hidden_layer.bias")?;
    let hidden_layer = Linear::new(hidden_weight, Some(hidden_bias));

    let output_weight = vb.get(
        (config.hidden_size, config.intermediate_size),
        "embed_layer.output_layer.weight",
    )?;
    let output_layer = Linear::new(output_weight, None);

    let res_weight = vb.get(
        (config.hidden_size, input_dim),
        "embed_layer.residual_layer.weight",
    )?;
    let res_bias = vb.get(config.hidden_size, "embed_layer.residual_layer.bias")?;
    let residual_layer = Linear::new(res_weight, Some(res_bias));

    println!("Weights loaded successfully");

    // Run forward pass
    let x = &input;
    let (batch_size, seq_len) = x.dims2()?;
    let patch_len = config.input_token_len;
    let step = config.input_token_len;
    let num_patches = seq_len / patch_len;

    println!("\nForward pass:");
    println!(
        "  batch_size={}, seq_len={}, patch_len={}, num_patches={}",
        batch_size, seq_len, patch_len, num_patches
    );

    // Pad if needed
    let padding = (patch_len - (seq_len % patch_len)) % patch_len;
    let x_padded = if padding == 0 {
        x.clone()
    } else {
        let pad_shape = vec![batch_size, padding];
        let zeros = Tensor::zeros(pad_shape, DType::F32, x.device())?;
        Tensor::cat(&[x, &zeros], 1)?
    };

    let (_, padded_len) = x_padded.dims2()?;
    println!("  x_padded shape: {:?}", x_padded.dims());

    // Create mask
    let mask = Tensor::ones((batch_size, padded_len), DType::F32, x.device())?;

    // Unfold
    let x_patches = unfold_patches(&x_padded, patch_len, step)?;
    let mask_patches = unfold_patches(&mask, patch_len, step)?;

    let (_, num_patches_out, patch_len_out) = x_patches.dims3()?;
    println!("  x_patches shape: {:?}", x_patches.dims());
    println!("  mask_patches shape: {:?}", mask_patches.dims());

    // Reshape
    let x_reshaped = x_patches.reshape((batch_size * num_patches_out, patch_len_out))?;
    let mask_reshaped = mask_patches.reshape((batch_size * num_patches_out, patch_len_out))?;

    // Concatenate
    let concatenated = Tensor::cat(&[&x_reshaped, &mask_reshaped], 1)?;
    println!("  concatenated shape: {:?}", concatenated.dims());

    // Save concatenated for comparison
    save_tensor_to_bin("/tmp/x_concat_rust.bin", &concatenated)?;

    // Apply hidden layer
    let hid = hidden_layer.forward(&concatenated)?;
    let hid_silu = candle_nn::ops::sigmoid(&hid)?.mul(&hid)?; // SiLU
    println!("  hid shape: {:?}", hid_silu.dims());

    // Save hid
    save_tensor_to_bin("/tmp/hid_rust.bin", &hid_silu)?;

    // Apply output layer
    let out = output_layer.forward(&hid_silu)?;
    println!("  out shape: {:?}", out.dims());

    // Apply residual
    let res = residual_layer.forward(&concatenated)?;
    let out = out.broadcast_add(&res)?;
    println!("  out after residual shape: {:?}", out.dims());

    // Reshape back
    let (_, hidden_size) = out.dims2()?;
    let result = out.reshape((batch_size, num_patches_out, hidden_size))?;
    println!("  final output shape: {:?}", result.dims());

    // Save output
    save_tensor_to_bin("/tmp/patch_embed_output_rust.bin", &result)?;

    // Compare with Python
    let rust_data: Vec<f32> = result.flatten_all()?.to_vec1()?;
    let python_data: Vec<f32> = python_output.flatten_all()?.to_vec1()?;

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for (r, p) in rust_data.iter().zip(python_data.iter()) {
        let diff = (r - p).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff;
    }
    let mean_diff = sum_diff / rust_data.len() as f32;

    println!("\n=== Comparison ===");
    println!("Max diff: {:.8e}", max_diff);
    println!("Mean diff: {:.8e}", mean_diff);

    // Check first few values
    println!("\nFirst 5 values:");
    for i in 0..5.min(rust_data.len()) {
        println!("  Rust:   {:>12.8e}", rust_data[i]);
        println!("  Python: {:>12.8e}", python_data[i]);
        println!("  Diff:   {:>12.8e}", (rust_data[i] - python_data[i]).abs());
    }

    Ok(())
}

fn unfold_patches(x: &Tensor, patch_len: usize, step: usize) -> Result<Tensor> {
    let (batch, seq_len) = x.dims2()?;
    let num_patches = (seq_len - patch_len) / step + 1;

    let mut patches = Vec::with_capacity(batch * num_patches * patch_len);

    for b in 0..batch {
        for p in 0..num_patches {
            let start = p * step;
            for i in 0..patch_len {
                let val = x.get(b)?.get(start + i)?.to_scalar::<f32>()?;
                patches.push(val);
            }
        }
    }

    Ok(Tensor::from_vec(
        patches,
        (batch, num_patches, patch_len),
        x.device(),
    )?)
}

fn load_npy(path: &str) -> Result<Tensor> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let header_len = u16::from_le_bytes([buffer[8], buffer[9]]) as usize;
    let header_start = 10;
    let header = std::str::from_utf8(&buffer[header_start..header_start + header_len])?;

    let shape = parse_shape_from_header(header)?;

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
