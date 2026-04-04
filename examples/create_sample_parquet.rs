use polars::prelude::*;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let timestamps = Series::new(
        "timestamp".into(),
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
            "2024-01-07",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ],
    );
    let values = Series::new(
        "value".into(),
        [
            100.5, 101.2, 102.8, 101.9, 103.5, 104.1, 105.0, 104.5, 106.2, 107.1,
        ],
    );
    let df = DataFrame::new(vec![Column::from(timestamps), Column::from(values)])?;

    let file = File::create("test_data/sample.parquet")?;
    ParquetWriter::new(file).finish(&mut df.clone())?;
    println!("Created test_data/sample.parquet");
    Ok(())
}
