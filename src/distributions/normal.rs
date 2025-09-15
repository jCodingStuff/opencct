//! Normal distribution

use std::f64::consts::{SQRT_2, PI};
use std::time::Duration;
use rand::{
    rngs::StdRng,
    Rng,
    SeedableRng,
};

use crate::{
    time::{DurationExtension, TimeUnit},
    Float,
};
use super::Distribution;

/// Box-Muller Transform
/// # Arguments
/// * `u`, `v` - Independent random numbers distributed uniformly on [0, 1]
/// # Returns
/// Two random variables that have the standard normal distribution
pub fn box_muller_transform(u: Float, v: Float) -> (Float, Float) {
    let term1 = SQRT_2 as Float * (-u.max(Float::EPSILON).ln()).sqrt();
    let term2 = 2.0 * PI as Float * v;
    (term1 * term2.cos(), term1 * term2.sin())
}

/// Scaled Box-Muller Transform
/// # Arguments
/// * `u`, `v` - Independent random numbers distributed uniformly on [0, 1]
/// * `mu` - Mean of the desired normal distribution
/// * `sigma` - Standard deviation of the desired normal distribution
/// # Returns
/// Two random variables that have the normal distribution with mean `mu` and standard deviation `sigma`
pub fn scaled_box_muller_transform(u: Float, v: Float, mu: Float, sigma: Float) -> (Float, Float) {
    let (x, y) = box_muller_transform(u, v);
    (mu + sigma * x, mu + sigma * y)
}

/// Normal distribution. Since in the current context, negative time does not make sense, the negative values
/// will be clamped to 0.
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, Normal};
/// use opencct::time::TimeUnit;
///
/// let mut dist = Normal::new(5.0, 1.0, TimeUnit::Millis);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct Normal {
    /// The mean (>= 0)
    mu      : Float,
    /// The standard deviation (> 0)
    sigma   : Float,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,

    /// Cached random variate
    cache   : Option<Float>,
}

impl Normal {
    /// Create a new [Normal] distribution with given mean and standard deviation.
    /// # Arguments
    /// * `mu` - Mean
    /// * `sigma` - Standard deviation
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [Normal].
    /// # Panic
    /// This function panics if `sigma <= 0`
    pub fn new(mu: Float, sigma: Float, unit: TimeUnit) -> Self {
        assert!(sigma > 0.0, "Sigma ({sigma}) must be > 0");
        Self { mu, sigma, factor: unit.factor(), rng: StdRng::from_os_rng(), cache: None }
    }

    /// Create a new [Normal] distribution with a specified random seed.
    /// # Arguments
    /// * `mu` - Mean
    /// * `sigma` - Standard deviation
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// A new [Normal].
    /// # Panic
    /// This function panics if `sigma <= 0`
    pub fn new_seeded(mu: Float, sigma: Float, unit: TimeUnit, seed: u64) -> Self {
        assert!(sigma > 0.0, "Sigma ({sigma}) must be > 0");
        Self { mu, sigma, factor: unit.factor(), rng: StdRng::seed_from_u64(seed), cache: None }
    }
}

impl Distribution for Normal {
    fn sample(&mut self, _: Duration) -> Duration {
        if let Some(variate) = self.cache.take() {
            return Duration::from_secs_float( variate * self.factor);
        }
        let (x, y) = scaled_box_muller_transform(
            self.rng.random::<Float>(),
            self.rng.random::<Float>(),
            self.mu,
            self.sigma,
        );
        self.cache = Some(y.max(0.0));
        Duration::from_secs_float(x.max(0.0) * self.factor)
    }
}

/// Normal distribution with time-varying parmeters. Since in the current context,
/// negative time does not make sense, the negative values will be clamped to 0.
/// Cache is not used because it is extremely unlikely to get two requests for the same time `t``.
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, NormalTV};
/// use opencct::time::{TimeUnit, DurationExtension};
///
/// let mut dist = NormalTV::new(|t| 1.0 + t.as_secs_float() * 0.1, |t| 3.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct NormalTV<FMu, FSigma> {
    /// The mean as a function of time
    mu      : FMu,
    /// The standard deviation as a function of time
    sigma   : FSigma,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
}

impl<FMu, FSigma> NormalTV<FMu, FSigma>
where
    FMu     : Fn(Duration) -> Float,
    FSigma  : Fn(Duration) -> Float,
{
    /// Create a new [NormalTV] distribution with given mean and standard deviation functions.
    /// # Arguments
    /// * `mu` - Function to compute the mean at a given time.
    /// * `sigma` - Function to compute the standard deviation at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// A new [NormalTV].
    /// # Be careful!
    /// `sigma` bound is not checked in release mode! Make sure you fulfill it!
    pub fn new(mu: FMu, sigma: FSigma, unit: TimeUnit) -> Self {
        Self { mu, sigma, factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    /// Create a new [NormalTV] distribution with a specified random seed.
    /// # Arguments
    /// * `mu` - Function to compute the mean at a given time.
    /// * `sigma` - Function to compute the standard deviation at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// A new [NormalTV].
    /// # Be careful!
    /// `sigma` bound is not checked in release mode! Make sure you fulfill it!
    pub fn new_seeded(mu: FMu, sigma: FSigma, unit: TimeUnit, seed: u64) -> Self {
        Self { mu, sigma, factor: unit.factor(), rng: StdRng::seed_from_u64(seed) }
    }
}

impl<FMu, FSigma> Distribution for NormalTV<FMu, FSigma>
where
    FMu: Fn(Duration) -> Float,
    FSigma: Fn(Duration) -> Float,
{
    /// See [Distribution::sample]
    /// # Panic
    /// In debug, this function will panic if at the requested time the standard deviation <= 0
    /// **This is NOT checked in release mode!**
    fn sample(&mut self, at: Duration) -> Duration {
        let (mu, sigma) = ((self.mu)(at), (self.sigma)(at));
        debug_assert!(sigma > 0.0, "Invalid sigma at {at:?}: {sigma}");
        let (x, _) = scaled_box_muller_transform(
            self.rng.random::<Float>(),
            self.rng.random::<Float>(),
            mu,
            sigma,
        );
        Duration::from_secs_float(x.max(0.0) * self.factor)
    }
}

#[cfg(test)]
mod box_muller_tests {
    use super::*;

    #[test]
    fn returns_two_values() {
        let (x, y) = box_muller_transform(0.5, 0.25);
        // Should return finite numbers
        assert!(x.is_finite(), "x should be finite, got {}", x);
        assert!(y.is_finite(), "y should be finite, got {}", y);
    }

    #[test]
    fn scaled_returns_with_mean_and_sigma() {
        let mu = 10.0;
        let sigma = 2.0;
        let (x, y) = scaled_box_muller_transform(0.5, 0.25, mu, sigma);
        // Roughly check that results are shifted/scaled
        assert!(x.is_finite() && y.is_finite());
        // Both should be "around" mu, but not exactly equal
        assert!((x - mu).abs() < 10.0, "x too far from mu: {}", x);
        assert!((y - mu).abs() < 10.0, "y too far from mu: {}", y);
    }

    #[test]
    fn reproducibility_same_inputs_same_outputs() {
        let (x1, y1) = box_muller_transform(0.123, 0.456);
        let (x2, y2) = box_muller_transform(0.123, 0.456);
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);

        let (sx1, sy1) = scaled_box_muller_transform(0.123, 0.456, 5.0, 2.0);
        let (sx2, sy2) = scaled_box_muller_transform(0.123, 0.456, 5.0, 2.0);
        assert_eq!(sx1, sx2);
        assert_eq!(sy1, sy2);
    }

    #[test]
    fn no_nan_with_small_u() {
        // Very small u should not produce NaN (clamped with EPSILON)
        let (x, y) = box_muller_transform(Float::MIN_POSITIVE, 0.75);
        assert!(x.is_finite(), "Expected finite x, got {}", x);
        assert!(y.is_finite(), "Expected finite y, got {}", y);
    }

    #[test]
    fn statistical_mean_and_variance_scaled() {
        // Law of large numbers check (approximation)
        let mu = 3.0;
        let sigma = 2.0;
        let n_samples = 50_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..n_samples {
            let u: Float = rand::random();
            let v: Float = rand::random();
            let (x, _) = scaled_box_muller_transform(u, v, mu, sigma);
            sum += x;
            sum_sq += x * x;
        }

        let mean = sum / n_samples as Float;
        let variance = sum_sq / n_samples as Float - mean * mean;

        assert!((mean - mu).abs() < 0.1, "mean {} not close to {}", mean, mu);
        assert!((variance - sigma * sigma).abs() < 0.2, "variance {} not close to {}", variance, sigma * sigma);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_positive() {
        let mut dist = Normal::new(5.0, 2.0, TimeUnit::Millis);

        for _ in 0..100 {
            let sample = dist.sample_at_t0();
            assert!(sample.as_secs_float() >= 0.0, "Normal sample should be >= 0, got {:?}", sample);
        }
    }

    #[test]
    fn seeded_reproducible() {
        let mu = 5.0;
        let sigma = 2.0;
        let seed = 42;
        let mut dist1 = Normal::new_seeded(mu, sigma, TimeUnit::Millis, seed);
        let mut dist2 = Normal::new_seeded(mu, sigma, TimeUnit::Millis, seed);

        for _ in 0..100 {
            let val1 = dist1.sample_at_t0();
            let val2 = dist2.sample_at_t0();
            assert_eq!(val1, val2, "Values {:?} and {:?} should be equal with the same seed", val1, val2);
        }
    }

    #[test]
    #[should_panic]
    fn invalid_sigma_panics() {
        let _ = Normal::new(5.0, 0.0, TimeUnit::Millis); // sigma <= 0 should panic
    }

    #[test]
    #[ignore]
    fn mean_approximation() {
        // Law of large numbers: mean ~ mu
        let mu = 5.0;
        let sigma = 2.0;
        let mut dist = Normal::new(mu, sigma, TimeUnit::Millis);
        let n_samples = 100_000;
        let mut sum = 0.0;
        for _ in 0..n_samples {
            sum += dist.sample_at_t0().as_millis_float();
        }
        let mean = sum / n_samples as Float;
        assert!((mean - mu).abs() < 0.05, "Sample mean {} not close to expected {}", mean, mu);
    }
}

#[cfg(test)]
mod tests_tv {
    use super::*;
    use std::time::Duration;

    #[test]
    fn samples_positive() {
        let mut dist = NormalTV::new(
            |t| 5.0 + t.as_secs_float() * 0.1,
            |t| 2.0 + t.as_secs_float() * 0.05,
            TimeUnit::Millis
        );

        for i in 0..10 {
            let t = Duration::from_secs(i);
            let sample = dist.sample(t);
            assert!(sample.as_secs_float() >= 0.0, "NormalTV sample should be >= 0, got {:?} at t={:?}", sample, t);
        }
    }

    #[test]
    fn seeded_reproducible() {
        let seed = 123;
        let mut dist1 = NormalTV::new_seeded(|_| 5.0, |_| 2.0, TimeUnit::Millis, seed);
        let mut dist2 = NormalTV::new_seeded(|_| 5.0, |_| 2.0, TimeUnit::Millis, seed);

        for _ in 0..100 {
            let val1 = dist1.sample_at_t0();
            let val2 = dist2.sample_at_t0();
            assert_eq!(val1, val2, "Values {:?} and {:?} should be equal with the same seed", val1, val2);
        }
    }

    #[test]
    #[ignore]
    fn mean_time_varying() {
        let mut dist = NormalTV::new(
            |t| 5.0 + t.as_secs_float() * 0.1,
            |t| 2.0 + t.as_secs_float() * 0.05,
            TimeUnit::Nanos
        );

        for t_sec in [0, 5, 10] {
            let t = Duration::from_secs(t_sec);
            let mu = 5.0 + t_sec as Float * 0.1;
            let mut sum = 0.0;
            let n_samples = 10_000;
            for _ in 0..n_samples {
                sum += dist.sample(t).as_nanos_float();
            }
            let mean = sum / n_samples as Float;
            assert!((mean - mu).abs() < 0.05, "At t={}, mean {} not close to expected {}", t_sec, mean, mu);
        }
    }
}
