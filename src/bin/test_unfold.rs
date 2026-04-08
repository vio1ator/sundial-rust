use candle_core::{Device, Tensor};
use std::fs::File;
use std::io::{Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Load input
    let input_data = load_npy("tests/reference_data/input.npy")?;
    println!("Input shape: {:?}", input_data.0);

    // Squeeze if needed
    let (batch, seq_len) = if input_data.0.len() == 3 {
        (input_data.0[0], input_data.0[1])
    } else {
        (input_data.0[0], input_data.0[1])
    };

    let input = Tensor::from_vec(input_data.1, (batch, seq_len), &device)?;

    // Apply Rust unfold
    let patch_len = 16;
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

    println!("Patches shape: {:?}", patches_tensor.dims());

    // Get first patch
    let first_patch = patches_tensor.get(0)?.get(0)?;
    let first_patch_vec: Vec<f32> = first_patch.to_vec1()?;
    println!("First patch first 5 values: {:?}", &first_patch_vec[..5]);

    // Save to binary
    save_tensor_to_bin("unfold_patches", &patches_tensor)?;

    Ok(())
}

fn load_npy(path: &str) -> Result<(Vec<usize>, Vec<f32>, usize), Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Parse numpy header
    let header_start = 10;
    let header_end = buffer[8..10]
        .iter()
        .fold(0, |acc, b| acc * 256 + (*b as usize));

    let header = std::str::from_utf8(&buffer[header_start..header_start + header_end])?;

    // Extract shape (simple parsing for 2D/3D arrays)
    let shape_start = header.find("'shape':").unwrap_or(0) + 9;
    let shape_paren = header[shape_start..].find('(').unwrap_or(0) + shape_start;
    let shape_end = header[shape_paren..].find(')').unwrap_or(0) + shape_paren;
    let shape_str = &header[shape_paren + 1..shape_end];

    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // Extract data (f32, little-endian)
    let data_start = header_start + header_end;
    let data = &buffer[data_start..];

    let data_vec: Vec<f32> = data
        .chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok((shape, data_vec, data_start))
}

fn save_tensor_to_bin(name: &str, tensor: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(format!("/tmp/{}.bin", name))?;
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

    let dims = tensor.dims();
    let dim_count = dims.len() as u32;
    file.write_all(&dim_count.to_le_bytes())?;
    for &d in dims {
        file.write_all(&(d as u32).to_le_bytes())?;
    }

    for &val in &data {
        file.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}
