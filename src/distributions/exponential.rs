//! Exponential distribution

use std::time::Duration;
use rand::{
    rngs::StdRng,
    Rng,
    SeedableRng,
};

use crate::Float;
use super::Distribution;

/// Exponential distribution.
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, Exponential};
///
/// let mut dist = Exponential::new(1.0);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {}", sample);
/// ```
pub struct Exponential {
    /// The rate parameter (> 0)
    lambda  : Float,
    /// Random number generator
    rng     : StdRng,
}

impl Exponential {
    /// Create a new [Exponential] distribution with given rate parameter.
    /// # Arguments
    /// * `lambda` - Rate parameter
    /// # Returns
    /// * A new [Exponential].
    /// # Panic
    /// This function panics if `lambda <= 0`
    pub fn new(lambda: Float) -> Self {
        assert!(lambda > 0.0, "Lambda ({lambda}) must be > 0");
        Self { lambda, rng: StdRng::from_os_rng() }
    }

    /// Create a new [Exponential] distribution with a specified random seed.
    /// # Arguments
    /// * `lambda` - Rate parameter
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// A new [Exponential].
    /// # Panic
    /// This function panics if `lambda <= 0`
    pub fn new_seeded(lambda: Float, seed: u64 ) -> Self {
        assert!(lambda > 0.0, "Lambda ({lambda}) must be > 0");
        Self { lambda, rng: StdRng::seed_from_u64(seed) }
    }
}

impl Distribution for Exponential {
    fn sample(&mut self, _: Duration) -> Float {
        - self.rng.random::<Float>().ln() / self.lambda
    }
}

/// Exponential distribution with time-varying rate parameter.
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, ExponentialTV};
/// use opencct::DurationExtension;
///
/// let mut dist = ExponentialTV::new(|t| 1.0 + t.as_secs_float() * 0.1);
/// let sample = dist.sample(Duration::from_secs(10));
/// println!("Sampled value: {}", sample);
/// ```
pub struct ExponentialTV<F> {
    /// Rate parameter as a function of time
    lambda   : F,
    /// Random number generator
    rng     : StdRng,
}

impl<F> ExponentialTV<F>
where
    F: Fn(Duration) -> Float,
{
    /// Create a new [ExponentialTV] distribution with a rate parameter function.
    /// # Arguments
    /// * `lambda` - Function to compute the rate parameter at a given time. Must be > 0 for any t >= 0
    /// # Returns
    /// A new [ExponentialTV].
    /// # Be careful!
    /// `lambda` is not checked in release mode! Make sure you fulfill the bounds!
    pub fn new(lambda: F) -> Self {
        Self { lambda, rng: StdRng::from_os_rng() }
    }

    /// Create a new [ExponentialTV] distribution with a specified random seed.
    /// # Arguments
    /// * `lambda` - Function to compute the rate parameter at a given time. Must be > 0 for any t >= 0
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// A new [ExponentialTV].
    /// # Be careful!
    /// `lambda` is not checked in release mode! Make sure you fulfill the bounds!
    pub fn new_seeded(lambda: F, seed: u64 ) -> Self {
        Self { lambda, rng: StdRng::seed_from_u64(seed) }
    }
}

impl<F> Distribution for ExponentialTV<F>
where
    F: Fn(Duration) -> Float,
{
    /// See [Distribution::sample]
    /// # Panic
    /// In debug, this function will panic if at the requested time the rate parameter <= 0.
    /// **This is NOT checked in release mode!**
    fn sample(&mut self, at: Duration) -> Float {
        let lambda = (self.lambda)(at);
        debug_assert!(lambda > 0.0, "Invalid lambda at {at:?}: {lambda}");
        - self.rng.random::<Float>().ln() / lambda
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_positive() {
        let mut dist = Exponential::new(1.0);

        for _ in 0..100 {
            let value = dist.sample_at_t0();
            assert!(value >= 0.0, "Exponential sample should be >= 0, got {}", value);
        }
    }

    #[test]
    fn seeded_reproducible() {
        let lambda = 2.0;
        let seed = 42;
        let mut dist1 = Exponential::new_seeded(lambda, seed);
        let mut dist2 = Exponential::new_seeded(lambda, seed);

        for _ in 0..100 {
            let val1 = dist1.sample_at_t0();
            let val2 = dist2.sample_at_t0();
            assert_eq!(val1, val2, "Values {} and {} should be equal with the same seed", val1, val2);
        }
    }

    #[test]
    #[should_panic]
    fn invalid_lambda_panics() {
        let _ = Exponential::new(0.0); // lambda <= 0 should panic
    }

    #[test]
    #[ignore]
    fn mean_approximation() {
        // Law of large numbers: mean ~ 1/lambda
        let lambda = 2.0;
        let mut dist = Exponential::new(lambda);
        let samples: Vec<Float> = (0..100_000).map(|_| dist.sample_at_t0()).collect();
        let mean: Float = samples.iter().sum::<Float>() / samples.len() as Float;
        assert!((mean - 1.0/lambda).abs() < 0.01, "Sample mean {} not close to expected {}", mean, 1.0/lambda);
    }
}

#[cfg(test)]
mod tests_tv {
    use crate::DurationExtension;
    use super::*;
    use std::time::Duration;

    #[test]
    fn samples_positive() {
        let mut dist = ExponentialTV::new(|t| 1.0 + t.as_secs_float() * 0.1);

        for i in 0..10 {
            let t = Duration::from_secs(i);
            let value = dist.sample(t);
            assert!(value >= 0.0, "ExponentialTV sample should be >= 0, got {} at t={:?}", value, t);
        }
    }

    #[test]
    fn seeded_reproducible() {
        let seed = 123;
        let mut dist1 = ExponentialTV::new_seeded(|_| 1.5, seed);
        let mut dist2 = ExponentialTV::new_seeded(|_| 1.5, seed);

        for _ in 0..100 {
            let val1 = dist1.sample_at_t0();
            let val2 = dist2.sample_at_t0();
            assert_eq!(val1, val2, "Values {} and {} should be equal with the same seed", val1, val2);
        }
    }

    #[test]
    #[should_panic]
    fn invalid_lambda_panics() {
        let mut dist = ExponentialTV::new(|_| 0.0);
        dist.sample_at_t0(); // lambda <= 0 should panic in debug mode
    }

    #[test]
    #[ignore]
    fn mean_time_varying() {
        let mut dist = ExponentialTV::new(|t| 1.0 + t.as_secs_float() * 0.1);
        // sample at t=0, t=5, t=10
        for t_sec in [0, 5, 10] {
            let t = Duration::from_secs(t_sec);
            let lambda = 1.0 + t_sec as Float * 0.1;
            let mut sum = 0.0;
            for _ in 0..10_000 {
                sum += dist.sample(t);
            }
            let mean = sum / 10_000.0;
            assert!((mean - 1.0/lambda).abs() < 0.02, "At t={}, mean {} not close to expected {}", t_sec, mean, 1.0/lambda);
        }
    }
}
