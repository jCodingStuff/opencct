//! Utilities for testing.
//!
//! This module provides helper functions commonly used in distribution
//! tests, such as computing descriptive statistics from samples and
//! asserting approximate equality with percentage-based tolerances.
//!
//! These utilities are not intended for use in production code but
//! only in `#[cfg(test)]` contexts across different distribution modules.

use crate::Float;

/// Compute the **sample mean** and **sample variance** of a slice of values.
///
/// # Arguments
/// * `samples` – A slice of sampled values.
///
/// # Returns
/// A tuple `(mean, variance)` computed as:
/// - `mean = Σx / n`
/// - `variance = Σ(x - mean)² / n`
///
/// # Example
/// ```
/// use opencct::test_utils::basic_statistics;
///
/// let samples = vec![1.0, 2.0, 3.0];
/// let (mean, var) = basic_statistics(&samples);
/// assert_eq!(mean, 2.0);
/// assert_eq!(var, 2.0/3.0);
/// ```
pub fn basic_statistics(samples: &[Float]) -> (Float, Float) {
    let mean = samples.iter().sum::<Float>() / samples.len() as Float;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<Float>() / samples.len() as Float;
    (mean, var)
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
        let (mean, var) = basic_statistics(&samples);

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
