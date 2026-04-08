use std::fs::File;
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::open("tests/reference_data/intermediates/patch_embed_output.npy")?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    println!(
        "Bytes 6-9: [{}, {}, {}, {}]",
        buffer[6], buffer[7], buffer[8], buffer[9]
    );

    // Version is bytes 6-7, header len is bytes 8-9
    let version = u16::from_le_bytes([buffer[6], buffer[7]]);
    let header_len = u16::from_le_bytes([buffer[8], buffer[9]]) as usize;
    println!("Version: {}, Header length: {}", version, header_len);

    let header_start = 10;
    let header = std::str::from_utf8(&buffer[header_start..header_start + header_len])?;
    println!("Header: {}", header);
    println!("Contains 'shape': {}", header.contains("shape"));

    Ok(())
}
