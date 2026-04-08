use candle_core::{Device, Tensor};
use std::fs::File;
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Load input
    let mut file = File::open("tests/reference_data/input.npy")?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Parse NPY
    let header_len = u16::from_le_bytes([buffer[8], buffer[9]]) as usize;
    let header = std::str::from_utf8(&buffer[10..10 + header_len])?;
    let shape = parse_shape(header)?;
    let data_start = 10 + header_len;
    let data: Vec<f32> = buffer[data_start..]
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Squeeze if 3D
    let input = if shape.len() == 3 {
        Tensor::from_vec(data, (shape[0], shape[1]), &device)?
    } else {
        Tensor::from_vec(data, shape.as_slice(), &device)?
    };

    println!("Input shape: {:?}", input.dims());

    // Apply Rust unfold
    let patch_len = 16;
    let (batch, seq_len) = input.dims2()?;
    let num_patches = seq_len / patch_len;

    let mut patches = Vec::with_capacity(batch * num_patches * patch_len);

    for b in 0..batch {
        for p in 0..num_patches {
            let start = p * patch_len;
            for i in 0..patch_len {
                let val = input.get(b)?.get(start + i)?.to_scalar::<f32>()?;
                patches.push(val);
            }
        }
    }

    let patches_tensor = Tensor::from_vec(patches, (batch, num_patches, patch_len), &device)?;

    println!("Rust patches shape: {:?}", patches_tensor.dims());

    // Get first patch
    let first_patch = patches_tensor.get(0)?.get(0)?;
    let first_patch_vec: Vec<f32> = first_patch.to_vec1()?;
    println!(
        "Rust first patch first 5 values: {:?}",
        &first_patch_vec[..5]
    );

    Ok(())
}

fn parse_shape(header: &str) -> anyhow::Result<Vec<usize>> {
    let shape_idx = header
        .find("shape")
        .ok_or_else(|| anyhow::anyhow!("No shape"))?;
    let rest = &header[shape_idx..];
    let paren_start = rest.find('(').ok_or_else(|| anyhow::anyhow!("No paren"))?;
    let paren_end = rest[paren_start..]
        .find(')')
        .ok_or_else(|| anyhow::anyhow!("No close paren"))?
        + paren_start;
    let shape_str = &rest[paren_start + 1..paren_end];
    Ok(shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect())
}
