//! Data loading and preprocessing for time series forecasting

use chrono::{DateTime, Datelike, Duration, Utc, Weekday};
use serde::{Deserialize, Serialize};

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
}

/// Time series dataset
#[derive(Debug, Clone)]
pub struct TimeSeriesData {
    pub timestamps: Vec<DateTime<Utc>>,
    pub values: Vec<f64>,
}

impl TimeSeriesData {
    /// Create new time series from vectors
    pub fn new(timestamps: Vec<DateTime<Utc>>, values: Vec<f64>) -> Self {
        assert_eq!(timestamps.len(), values.len());
        Self { timestamps, values }
    }

    /// Get length of series
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if series is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get last value
    pub fn last_value(&self) -> Option<f64> {
        self.values.last().copied()
    }

    /// Normalize data (z-score normalization)
    pub fn normalize(&self) -> (Vec<f64>, f64, f64) {
        let mean = self.values.iter().sum::<f64>() / self.values.len() as f64;
        let std = (self.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / self.values.len() as f64)
            .sqrt();

        let normalized: Vec<f64> = self.values.iter().map(|v| (v - mean) / std).collect();

        (normalized, mean, std)
    }

    /// Denormalize data
    pub fn denormalize(values: &[f64], mean: f64, std: f64) -> Vec<f64> {
        values.iter().map(|v| v * std + mean).collect()
    }

    /// Convert to tensor-compatible format
    pub fn to_tensor_format(&self) -> Vec<f64> {
        self.values.clone()
    }
}

/// Data loader for various sources
pub struct DataLoader;

impl DataLoader {
    /// Load SPY data from yfinance (would need to call API)
    pub fn load_spy(_period: &str) -> Result<TimeSeriesData, String> {
        // TODO: Implement actual yfinance API call
        // This would use reqwest to call a financial data API

        // Mock data for now
        let end = Utc::now();
        let start = end - Duration::days(365 * 3); // 3 years

        let mut timestamps = Vec::new();
        let mut values = Vec::new();

        let mut current = start;
        let mut value = 450.0;

        while current <= end {
            let weekday = current.weekday();
            if weekday != Weekday::Sat && weekday != Weekday::Sun {
                timestamps.push(current);
                values.push(value);
                // Random walk for demo
                value += (value * 0.001) * ((current.timestamp() as f64) % 10.0 - 5.0);
            }
            current += Duration::days(1);
        }

        Ok(TimeSeriesData::new(timestamps, values))
    }

    /// Create sliding window sequences
    pub fn create_windows(data: &[f64], window_size: usize) -> Vec<Vec<f64>> {
        data.windows(window_size).map(|w| w.to_vec()).collect()
    }
}

/// Configuration for data loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub ticker: String,
    pub period: String,
    pub interval: String,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            ticker: "SPY".to_string(),
            period: "3y".to_string(),
            interval: "1d".to_string(),
            start_date: None,
            end_date: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let data = TimeSeriesData::new(
            vec![Utc::now(); 10],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        );

        let (normalized, mean, std) = data.normalize();

        assert!((mean - 5.5).abs() < 1e-6);
        assert!((std - 2.872).abs() < 0.01);
        assert_eq!(normalized.len(), 10);
    }

    #[test]
    fn test_windows() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let windows = DataLoader::create_windows(&data, 3);

        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(windows[1], vec![2.0, 3.0, 4.0]);
        assert_eq!(windows[2], vec![3.0, 4.0, 5.0]);
    }
}
