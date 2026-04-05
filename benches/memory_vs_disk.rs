//! Benchmark memory vs disk weight loading performance
//!
//! This benchmark compares the startup performance of:
//! - Memory loading: Decompressing weights into memory and loading directly
//! - Disk loading: Extracting weights to disk first, then loading from file
//!
//! Run with: `cargo bench --features bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use candle_core::Device;
use std::time::Instant;

/// Benchmark the memory loading path using WeightLoader::load_into_candle()
/// This decompresses weights into memory and loads them directly without disk I/O.
fn benchmark_memory_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_loading");
    
    group.measurement_time(std::time::Duration::from_secs(1));
    group.sample_size(5);
    
    group.bench_function("load_into_candle", |b| {
        b.iter(|| {
            let device = Device::Cpu;
            
            // Create weight loader (this keeps weights in memory by default)
            let start = Instant::now();
            let loader = sundial_rust::weights::loader::WeightLoader::new_with_memory_weights()
                .expect("Failed to create weight loader with memory weights");
            
            // Load into candle - this decompresses and loads tensors
            let varbuilder = loader.load_into_candle(&device)
                .expect("Failed to load weights into candle");
            
            let elapsed = start.elapsed();
            
            // Verify we got a valid varbuilder
            black_box(varbuilder);
            
            // Print timing for visibility during benchmark run
            if elapsed.as_millis() < 1000 {
                eprintln!("Memory loading completed in {:.2?}ms", elapsed.as_micros() as f64 / 1000.0);
            } else {
                eprintln!("Memory loading completed in {:.2?}s", elapsed.as_secs_f64());
            }
        });
    });
    
    group.finish();
}

/// Benchmark the disk loading path using SUNDIAL_USE_DISK environment variable
/// This extracts weights to disk first, then loads from the file.
fn benchmark_disk_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("disk_loading");
    
    group.measurement_time(std::time::Duration::from_secs(1));
    group.sample_size(5);
    
    group.bench_function("load_from_disk", |b| {
        use std::env;
        
        b.iter(|| {
            // Set environment variable to force disk mode
            env::set_var("SUNDIAL_USE_DISK", "true");
            
            let device = Device::Cpu;
            
            // Create weight loader (this will extract to disk)
            let start = Instant::now();
            let loader_result = sundial_rust::weights::loader::WeightLoader::new();
            
            let loader = match loader_result {
                Ok(loader) => loader,
                Err(e) => {
                    // Skip this iteration if disk loading fails
                    eprintln!("Disk loading failed: {}, skipping iteration", e);
                    env::remove_var("SUNDIAL_USE_DISK");
                    return;
                }
            };
            
            // Load into candle from disk
            let varbuilder = loader.load_into_candle(&device)
                .expect("Failed to load weights from disk into candle");
            
            let elapsed = start.elapsed();
            
            // Clean up environment variable
            env::remove_var("SUNDIAL_USE_DISK");
            
            // Verify we got a valid varbuilder
            black_box(varbuilder);
            
            // Print timing for visibility during benchmark run
            if elapsed.as_millis() < 1000 {
                eprintln!("Disk loading completed in {:.2?}ms", elapsed.as_micros() as f64 / 1000.0);
            } else {
                eprintln!("Disk loading completed in {:.2?}s", elapsed.as_secs_f64());
            }
        });
    });
    
    group.finish();
}

/// Benchmark comparing both loading methods side by side
fn benchmark_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison");
    
    group.measurement_time(std::time::Duration::from_secs(3));
    group.sample_size(20);
    
    group.bench_function("memory_vs_disk_overhead", |b| {
        use std::env;
        
        b.iter(|| {
            let device = Device::Cpu;
            
            // Time memory loading
            let mem_start = Instant::now();
            let mem_loader = sundial_rust::weights::loader::WeightLoader::new_with_memory_weights()
                .expect("Failed to create memory weight loader");
            let _mem_varbuilder = mem_loader.load_into_candle(&device)
                .expect("Failed to load memory weights");
            let mem_time = mem_start.elapsed();
            
            // Time disk loading
            env::set_var("SUNDIAL_USE_DISK", "true");
            let disk_start = Instant::now();
            let disk_loader = match sundial_rust::weights::loader::WeightLoader::new() {
                Ok(loader) => loader,
                Err(_) => {
                    env::remove_var("SUNDIAL_USE_DISK");
                    return;
                }
            };
            let _disk_varbuilder = disk_loader.load_into_candle(&device)
                .expect("Failed to load disk weights");
            let disk_time = disk_start.elapsed();
            env::remove_var("SUNDIAL_USE_DISK");
            
            // Calculate overhead
            let overhead = if disk_time > mem_time {
                disk_time.as_secs_f64() / mem_time.as_secs_f64()
            } else {
                1.0
            };
            
            black_box(overhead);
            
            eprintln!(
                "Memory: {:.2?}ms, Disk: {:.2?}ms, Overhead: {:.2}x",
                mem_time.as_micros() as f64 / 1000.0,
                disk_time.as_micros() as f64 / 1000.0,
                overhead
            );
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_memory_loading,
    benchmark_disk_loading,
    benchmark_comparison,
);

criterion_main!(benches);
