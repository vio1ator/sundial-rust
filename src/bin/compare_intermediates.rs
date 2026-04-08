//! Compare Rust and Python intermediate tensors to find divergence point

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::fs::File;
use std::io::Read;

fn main() -> Result<()> {
    println!("=== Comparing Rust and Python Intermediates ===\n");

    // List of intermediates to compare
    let intermediates = vec![
        "patch_embed_output",
        "layer_0_output",
        "layer_1_output",
        "layer_2_output",
        "layer_3_output",
        "layer_4_output",
        "layer_5_output",
        "layer_6_output",
        "layer_7_output",
        "layer_8_output",
        "layer_9_output",
        "layer_10_output",
        "layer_11_output",
        "transformer_output",
        "predictions",
    ];

    let mut first_divergence = None;

    for name in intermediates {
        // Load Python reference
        let python_path = format!("tests/reference_data/intermediates/{}.npy", name);
        let python_tensor = load_npy(&python_path)?;

        // Load Rust output
        let rust_path = format!("/tmp/{}_rust.bin", name);
        let rust_tensor = match load_tensor_from_bin(&rust_path) {
            Ok(t) => t,
            Err(e) => {
                println!("[SKIP] {}: Rust file not found - {}", name, e);
                continue;
            }
        };

        // Compare
        let comparison = compare_tensors(&rust_tensor, &python_tensor, name)?;

        println!(
            "{}: max_diff={:.8e}, mean_diff={:.8e}",
            name, comparison.max_diff, comparison.mean_diff
        );

        if comparison.max_diff > 1e-5 && first_divergence.is_none() {
            first_divergence = Some(name.to_string());
        }
    }

    println!("\n=== Summary ===");
    match first_divergence {
        Some(name) => println!("First divergence detected at: {}", name),
        None => println!("No significant divergence detected"),
    }

    Ok(())
}

struct Comparison {
    max_diff: f32,
    mean_diff: f32,
}

fn compare_tensors(rust_tensor: &Tensor, python_tensor: &Tensor, name: &str) -> Result<Comparison> {
    let rust_data: Vec<f32> = rust_tensor.flatten_all()?.to_vec1()?;
    let python_data: Vec<f32> = python_tensor.flatten_all()?.to_vec1()?;

    if rust_data.len() != python_data.len() {
        println!(
            "[ERROR] {}: Shape mismatch - Rust: {}, Python: {}",
            name,
            rust_data.len(),
            python_data.len()
        );
        return Ok(Comparison {
            max_diff: f32::MAX,
            mean_diff: f32::MAX,
        });
    }

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

    Ok(Comparison {
        max_diff,
        mean_diff,
    })
}

fn load_npy(path: &str) -> Result<Tensor> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Parse numpy header
    // Bytes 0-5: magic string '\x93NUMPY'
    // Bytes 6-7: version
    // Bytes 8-9: header length (little-endian uint16 for v1.0)
    let header_len = u16::from_le_bytes([buffer[8], buffer[9]]) as usize;
    let header_start = 10;
    let header = std::str::from_utf8(&buffer[header_start..header_start + header_len])?;

    // Extract shape - parse 'shape': (1, 180, 768) format using simple string search
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
    // Parse 'shape': (1, 180, 768) format using simple string search
    let shape_idx = header
        .find("shape")
        .ok_or_else(|| anyhow::anyhow!("No shape in header"))?;
    let rest = &header[shape_idx..];

    // Find the opening paren after 'shape'
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

fn load_tensor_from_bin(path: &str) -> Result<Tensor> {
    use std::io::Read;

    let mut file = File::open(path).map_err(|e| anyhow::anyhow!("Failed to open file: {}", e))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| anyhow::anyhow!("Failed to read file: {}", e))?;

    // Read shape
    let mut pos = 0;
    if buffer.len() < 4 {
        return Err(anyhow::anyhow!("Buffer too short for dimension count"));
    }
    let dim_count = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]) as usize;
    pos += 4;

    let mut dims = Vec::with_capacity(dim_count);
    for _ in 0..dim_count {
        if pos + 4 > buffer.len() {
            return Err(anyhow::anyhow!("Buffer too short for dimensions"));
        }
        let d = u32::from_le_bytes([
            buffer[pos],
            buffer[pos + 1],
            buffer[pos + 2],
            buffer[pos + 3],
        ]) as usize;
        dims.push(d);
        pos += 4;
    }

    // Read data
    let data: Vec<f32> = buffer[pos..]
        .chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let device = candle_core::Device::Cpu;
    Ok(Tensor::from_vec(data, dims, &device)?)
}
