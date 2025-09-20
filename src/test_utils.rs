//! Utilities for testing.
//!
//! This module provides helper functions commonly used in distribution
//! tests, such as computing descriptive statistics from samples and
//! asserting approximate equality with percentage-based tolerances.
//!
//! These utilities are not intended for use in production code but
//! only in `#[cfg(test)]` contexts across different distribution modules.

use crate::Float;

/// A simple container for basic population statistics computed from a slice of samples.
///
/// Currently this struct stores the **mean**, **variance**, and provides a method
/// to compute the **standard deviation** of the provided sample data.
/// It is designed to be easily extended in the future (e.g., adding median,
/// skewness, or higher-order moments) without breaking existing code.
///
/// # Notes
/// - Both the mean and variance are computed as *population* statistics:
///   - Mean: `Σx / N`
///   - Variance: `Σ(x - mean)² / N`
/// - For *sample* variance and standard deviation (with `N - 1` in the denominator),
///   you would need to implement a different method.
///
/// # Examples
/// ```
/// use opencct::test_utils::BasicStatistics;
///
/// let samples = vec![1.0, 2.0, 3.0, 4.0];
/// let stats = BasicStatistics::compute(&samples);
///
/// assert_eq!(stats.mean(), 2.5);
/// assert_eq!(stats.variance(), 1.25);
/// assert!((stats.std() - 1.1180).abs() < 1e-4);
/// ```
pub struct BasicStatistics {
    mean        : Float,
    variance    : Float,
}

impl BasicStatistics {
    /// Compute mean and variance for the given slice of samples.
    ///
    /// # Panics
    /// This method panics if `samples` is empty.
    pub fn compute(samples: &[Float]) -> Self {
        let mean = samples.iter().sum::<Float>() / samples.len() as Float;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<Float>() / samples.len() as Float;
        Self { mean, variance }
    }

    /// Return the mean of the samples.
    pub fn mean(&self) -> Float { self.mean }

    /// Return the variance of the samples.
    pub fn variance(&self) -> Float { self.variance }

    /// Return the standard deviation of the samples.
    ///
    /// Computed as the square root of the population variance.
    pub fn std(&self) -> Float { self.variance.sqrt() }
}

/// Assert that two floating-point values are approximately equal within
/// a relative tolerance expressed as a percentage of the expected value.
///
/// # Arguments
/// * `actual` – The observed value.
/// * `expected` – The theoretical or expected value.
/// * `tolerance` – The maximum allowed relative error (e.g., `0.05` for 5%).
/// * `label` – A label for the assertion, included in failure messages.
///
/// # Panics
/// Panics if the absolute difference exceeds `expected * tolerance`.
///
/// # Example
/// ```
/// use opencct::test_utils::assert_close;
///
/// let actual = 10.2;
/// let expected = 10.0;
/// assert_close(actual, expected, 0.05, "test value"); // passes
/// ```
pub fn assert_close(actual: Float, expected: Float, tolerance: Float, label: &str) {
    assert!(
        (actual - expected).abs() <= expected * tolerance,
        "{label} {actual} outside tolerance of expected {expected}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_statistics_mean_and_variance() {
        let samples = [1.0, 2.0, 3.0, 4.0];
        let stats = BasicStatistics::compute(&samples);
        let (mean, var) = (stats.mean, stats.variance);

        // The mean should be 2.5
        assert!((mean - 2.5).abs() < 1e-12, "mean mismatch, got {mean}");

        // Population variance = Σ(x - mean)^2 / n = 1.25
        assert!((var - 1.25).abs() < 1e-12, "variance mismatch, got {var}");
    }

    #[test]
    fn test_assert_close_passes() {
        // Should not panic
        assert_close(10.2, 10.0, 0.05, "test_value");
    }

    #[test]
    #[should_panic]
    fn test_assert_close_fails() {
        // Should panic because difference is 6%, tolerance 5%
        assert_close(10.6, 10.0, 0.05, "test_value");
    }
}
