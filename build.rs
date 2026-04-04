use flate2::write::GzEncoder;
use flate2::Compression;
use std::env;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// Compute SHA256 hash of a file
fn compute_sha256<P: AsRef<Path>>(path: P) -> Result<String, String> {
    use sha2::{Digest, Sha256};

    let mut file = File::open(path.as_ref()).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer).map_err(|e| e.to_string())?;

        if bytes_read == 0 {
            break;
        }

        hasher.update(&buffer[..bytes_read]);
    }

    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

/// Compress the weights file using gzip
fn compress_weights(source_path: &Path, output_path: &Path) -> Result<(), BuildError> {
    println!(
        "cargo:warning=Compressing weights from {}...",
        source_path.display()
    );

    let mut input_file = File::open(source_path)
        .map_err(|e| BuildError::FileOpen(e.to_string(), source_path.to_path_buf()))?;

    let input_size = input_file
        .metadata()
        .map_err(|e| BuildError::FileMetadata(e.to_string(), source_path.to_path_buf()))?
        .len();

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());

    io::copy(&mut input_file, &mut encoder).map_err(|e| BuildError::Compression(e.to_string()))?;

    let compressed_data = encoder
        .finish()
        .map_err(|e| BuildError::Compression(e.to_string()))?;

    let output_size = compressed_data.len() as u64;

    let compression_ratio = input_size as f64 / output_size as f64;

    println!("cargo:warning=  Original size: {} bytes", input_size);
    println!("cargo:warning=  Compressed size: {} bytes", output_size);
    println!(
        "cargo:warning=  Compression ratio: {:.2}x",
        compression_ratio
    );

    std::fs::write(output_path, &compressed_data)
        .map_err(|e| BuildError::FileWrite(e.to_string(), output_path.to_path_buf()))?;

    println!(
        "cargo:warning=  Compressed weights written to: {}",
        output_path.display()
    );

    Ok(())
}

fn main() -> Result<(), BuildError> {
    // Detect target platform
    let target_os = env::var("CARGO_CFG_TARGET_OS")
        .map_err(|e| BuildError::Environment(format!("CARGO_CFG_TARGET_OS: {}", e)))?;
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")
        .map_err(|e| BuildError::Environment(format!("CARGO_CFG_TARGET_ARCH: {}", e)))?;
    let target_family = env::var("CARGO_CFG_TARGET_FAMILY")
        .map_err(|e| BuildError::Environment(format!("CARGO_CFG_TARGET_FAMILY: {}", e)))?;

    println!(
        "cargo:warning=Target platform: {}-{} ({})",
        target_os, target_arch, target_family
    );

    // Set platform-specific environment variables
    println!("cargo:rustc-env=TARGET_OS={}", target_os);
    println!("cargo:rustc-env=TARGET_ARCH={}", target_arch);
    println!("cargo:rustc-env=TARGET_FAMILY={}", target_family);

    // Generate platform-specific configuration
    let platform_config = generate_platform_config(&target_os, &target_arch, &target_family);
    let config_path = write_platform_config(&platform_config)?;
    println!(
        "cargo:warning=Platform config written to: {}",
        config_path.display()
    );

    // Get the source weights path
    let weights_dir = Path::new("weights");
    let source_weights = weights_dir.join("model.safetensors");

    if !source_weights.exists() {
        return Err(BuildError::MissingWeights(source_weights));
    }

    // Get the out_dir from OUT_DIR environment variable
    let out_dir = env::var("OUT_DIR")
        .map_err(|e| BuildError::Environment(format!("{}: {}", "OUT_DIR", e)))?;
    let out_path = Path::new(&out_dir);

    // Create a subdirectory for embedded weights
    let embedded_dir = out_path.join("embedded_weights");
    std::fs::create_dir_all(&embedded_dir)
        .map_err(|e| BuildError::Directory(e.to_string(), embedded_dir.clone()))?;

    // Output paths
    let compressed_path = embedded_dir.join("model.safetensors.gz");
    let hash_path = embedded_dir.join("model.safetensors.sha256");

    // Compute SHA256 hash of original weights
    let sha256_hash = compute_sha256(&source_weights).map_err(BuildError::Hash)?;

    println!("cargo:warning=  SHA256 hash: {}", sha256_hash);

    // Write hash to file for later use
    std::fs::write(&hash_path, format!("{}\n", sha256_hash))
        .map_err(|e| BuildError::FileWrite(e.to_string(), hash_path.clone()))?;

    // Compress weights
    compress_weights(&source_weights, &compressed_path)?;

    // Set environment variable for the build
    let weights_path = compressed_path
        .to_str()
        .ok_or(BuildError::PathConversion(compressed_path.clone()))?;

    println!(
        "cargo:warning=  WEIGHTS_PATH env var set to: {}",
        weights_path
    );
    println!("cargo:rerun-if-changed={}", source_weights.display());
    println!("cargo:rerun-if-changed={}", hash_path.display());

    // Emit the environment variable
    println!("cargo:rustc-env=WEIGHTS_PATH={}", weights_path);
    println!("cargo:rustc-env=MODEL_SHA256={}", sha256_hash);

    Ok(())
}

/// Generate platform-specific configuration
fn generate_platform_config(
    target_os: &str,
    target_arch: &str,
    target_family: &str,
) -> PlatformConfig {
    PlatformConfig {
        os: target_os.to_string(),
        arch: target_arch.to_string(),
        family: target_family.to_string(),
        is_unix: target_family == "unix",
        is_windows: target_os == "windows",
        is_macos: target_os == "macos",
        is_linux: target_os == "linux",
        is_x86: target_arch == "x86" || target_arch == "x86_64",
        is_arm: target_arch == "aarch64" || target_arch == "arm",
    }
}

/// Write platform configuration to a JSON file
fn write_platform_config(config: &PlatformConfig) -> Result<PathBuf, BuildError> {
    let out_dir =
        env::var("OUT_DIR").map_err(|e| BuildError::Environment(format!("OUT_DIR: {}", e)))?;
    let config_path = Path::new(&out_dir).join("platform_config.json");

    let config_json = serde_json::to_string_pretty(config)
        .map_err(|e| BuildError::Serialization(e.to_string()))?;

    std::fs::write(&config_path, config_json)
        .map_err(|e| BuildError::FileWrite(e.to_string(), config_path.clone()))?;

    Ok(config_path)
}

/// Platform-specific configuration structure
#[derive(Debug, serde::Serialize)]
struct PlatformConfig {
    os: String,
    arch: String,
    family: String,
    is_unix: bool,
    is_windows: bool,
    is_macos: bool,
    is_linux: bool,
    is_x86: bool,
    is_arm: bool,
}

/// Custom error types for build script
#[derive(Debug)]
enum BuildError {
    MissingWeights(PathBuf),
    FileOpen(String, PathBuf),
    FileMetadata(String, PathBuf),
    FileWrite(String, PathBuf),
    Compression(String),
    Directory(String, PathBuf),
    Environment(String),
    PathConversion(PathBuf),
    Hash(String),
    Serialization(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::MissingWeights(path) => {
                write!(f, "Weights file not found: {}", path.display())
            }
            BuildError::FileOpen(msg, path) => {
                write!(f, "Failed to open file {}: {}", path.display(), msg)
            }
            BuildError::FileMetadata(msg, path) => {
                write!(f, "Failed to get metadata for {}: {}", path.display(), msg)
            }
            BuildError::FileWrite(msg, path) => {
                write!(f, "Failed to write to {}: {}", path.display(), msg)
            }
            BuildError::Compression(msg) => {
                write!(f, "Compression failed: {}", msg)
            }
            BuildError::Directory(msg, path) => {
                write!(f, "Failed to create directory {}: {}", path.display(), msg)
            }
            BuildError::Environment(msg) => {
                write!(f, "Failed to read environment variable: {}", msg)
            }
            BuildError::PathConversion(path) => {
                write!(f, "Failed to convert path to string: {}", path.display())
            }
            BuildError::Hash(msg) => {
                write!(f, "Hash computation failed: {}", msg)
            }
            BuildError::Serialization(msg) => {
                write!(f, "Serialization failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for BuildError {}
