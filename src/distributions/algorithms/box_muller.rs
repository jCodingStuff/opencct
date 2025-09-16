//! # Box-Muller Transform
//! Box, G. E., & Muller, M. E. (1958).
//! [A note on the generation of random normal deviates](https://doi.org/10.1214%2Faoms%2F1177706645).
//! *The annals of mathematical statistics*, 29(2), 610-611.

use rand::Rng;
use std::f64::consts::{SQRT_2, PI};

use crate::Float;

/// Box-Muller Transform
/// # Arguments
/// * `rng` - A random number generator
/// # Returns
/// Two random variables that have the standard normal distribution
pub fn box_muller_transform<R: Rng + ?Sized>(rng: &mut R) -> (Float, Float) {
    let (u, v) = (rng.random::<Float>(), rng.random::<Float>());
    let term1 = SQRT_2 as Float * (-u.max(Float::EPSILON).ln()).sqrt();
    let term2 = 2.0 * PI as Float * v;
    (term1 * term2.cos(), term1 * term2.sin())
}

/// Scaled Box-Muller Transform
/// # Arguments
/// * `rng` - A random number generator
/// * `mu` - Mean of the desired normal distribution
/// * `sigma` - Standard deviation of the desired normal distribution
/// # Returns
/// Two random variables that have the normal distribution with mean `mu` and standard deviation `sigma`
pub fn scaled_box_muller_transform<R: Rng + ?Sized>(rng: &mut R, mu: Float, sigma: Float) -> (Float, Float) {
    let (x, y) = box_muller_transform(rng);
    (mu + sigma * x, mu + sigma * y)
}

#[cfg(test)]
mod box_muller_tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn outputs_are_finite() {
        let mut rng = StdRng::seed_from_u64(123);
        let (x, y) = box_muller_transform(&mut rng);
        assert!(x.is_finite(), "x should be finite, got {x}");
        assert!(y.is_finite(), "y should be finite, got {y}");
    }

    #[test]
    fn scaled_outputs_are_finite() {
        let mut rng = StdRng::seed_from_u64(123);
        let (x, y) = scaled_box_muller_transform(&mut rng, 5.0, 2.0);
        assert!(x.is_finite(), "x should be finite, got {x}");
        assert!(y.is_finite(), "y should be finite, got {y}");
    }

    #[test]
    fn reproducibility_same_seed() {
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);

        let (x1, y1) = box_muller_transform(&mut rng1);
        let (x2, y2) = box_muller_transform(&mut rng2);

        assert_eq!(x1, x2, "Deterministic output mismatch: {x1} vs {x2}");
        assert_eq!(y1, y2, "Deterministic output mismatch: {y1} vs {y2}");
    }

    #[test]
    #[ignore]
    fn statistical_mean_and_variance_scaled() {
        let mu = 30.0;
        let sigma = 20.0;
        let mut rng = StdRng::seed_from_u64(999);

        let n_samples = 50_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..n_samples {
            let (x, _) = scaled_box_muller_transform(&mut rng, mu, sigma);
            sum += x;
            sum_sq += x * x;
        }

        let mean = sum / n_samples as f64;
        let variance = sum_sq / n_samples as f64 - mean.powi(2);

        let mean_error = (mean - mu).abs() / mu.abs();
        let variance_error = (variance - sigma * sigma).abs() / (sigma * sigma);

        assert!(
            mean_error < 0.05,
            "mean {} not within 5% of expected {} (error {:.2}%)",
            mean,
            mu,
            mean_error * 100.0
        );

        assert!(
            variance_error < 0.1,
            "variance {} not within 10% of expected {} (error {:.2}%)",
            variance,
            sigma * sigma,
            variance_error * 100.0
        );
    }
}
