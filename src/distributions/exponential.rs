//! Exponential distribution

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

/// Exponential distribution.
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, Exponential};
/// use opencct::time::TimeUnit;
///
/// let mut dist = Exponential::new(1.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct Exponential {
    /// The rate parameter (> 0)
    lambda  : Float,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
}

impl Exponential {
    /// Create a new [Exponential] distribution with given rate parameter.
    /// # Arguments
    /// * `lambda` - Rate parameter
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [Exponential].
    /// # Panic
    /// This function panics if `lambda <= 0`
    pub fn new(lambda: Float, unit: TimeUnit) -> Self {
        assert!(lambda > 0.0, "Lambda ({lambda}) must be > 0");
        Self { lambda, factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    /// Create a new [Exponential] distribution with a specified random seed.
    /// # Arguments
    /// * `lambda` - Rate parameter
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// A new [Exponential].
    /// # Panic
    /// This function panics if `lambda <= 0`
    pub fn new_seeded(lambda: Float, unit: TimeUnit, seed: u64) -> Self {
        assert!(lambda > 0.0, "Lambda ({lambda}) must be > 0");
        Self { lambda, factor: unit.factor(), rng: StdRng::seed_from_u64(seed) }
    }
}

impl Distribution for Exponential {
    fn sample(&mut self, _: Duration) -> Duration {
        let raw = - self.rng.random::<Float>().ln() / self.lambda;
        Duration::from_secs_float(raw * self.factor)
    }
}

/// Exponential distribution with time-varying rate parameter.
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, ExponentialTV};
/// use opencct::time::{DurationExtension, TimeUnit};
///
/// let mut dist = ExponentialTV::new(|t| 1.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
/// let sample = dist.sample(Duration::from_secs(10));
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct ExponentialTV<F> {
    /// Rate parameter as a function of time
    lambda   : F,
    /// Time unit factor
    factor  : Float,
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
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// A new [ExponentialTV].
    /// # Be careful!
    /// `lambda` is not checked in release mode! Make sure you fulfill the bounds!
    pub fn new(lambda: F, unit: TimeUnit) -> Self {
        Self { lambda, factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    /// Create a new [ExponentialTV] distribution with a specified random seed.
    /// # Arguments
    /// * `lambda` - Function to compute the rate parameter at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// A new [ExponentialTV].
    /// # Be careful!
    /// `lambda` is not checked in release mode! Make sure you fulfill the bounds!
    pub fn new_seeded(lambda: F, unit: TimeUnit, seed: u64 ) -> Self {
        Self { lambda, factor: unit.factor(), rng: StdRng::seed_from_u64(seed) }
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
    fn sample(&mut self, at: Duration) -> Duration {
        let lambda = (self.lambda)(at);
        debug_assert!(lambda > 0.0, "Invalid lambda at {at:?}: {lambda}");
        let raw = - self.rng.random::<Float>().ln() / lambda;
        Duration::from_secs_float(raw * self.factor)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{TimeUnit, DurationExtension};

    #[test]
    fn samples_positive() {
        let mut dist = Exponential::new(1.0, TimeUnit::Seconds);

        for _ in 0..100 {
            let value = dist.sample_at_t0().as_secs_float();
            assert!(value >= 0.0, "Exponential sample should be >= 0, got {}", value);
        }
    }

    #[test]
    fn seeded_reproducible() {
        let lambda = 2.0;
        let seed = 42;
        let mut dist1 = Exponential::new_seeded(lambda, TimeUnit::Seconds, seed);
        let mut dist2 = Exponential::new_seeded(lambda, TimeUnit::Seconds, seed);

        for _ in 0..100 {
            let val1 = dist1.sample_at_t0().as_secs_float();
            let val2 = dist2.sample_at_t0().as_secs_float();
            assert_eq!(val1, val2, "Values {} and {} should be equal with the same seed", val1, val2);
        }
    }

    #[test]
    #[should_panic]
    fn invalid_lambda_panics() {
        let _ = Exponential::new(0.0, TimeUnit::Seconds);
    }

    #[test]
    #[ignore]
    fn mean_approximation() {
        let lambda = 2.0;
        let mut dist = Exponential::new(lambda, TimeUnit::Seconds);
        let samples: Vec<Float> = (0..100_000)
            .map(|_| dist.sample_at_t0().as_secs_float())
            .collect();
        let mean: Float = samples.iter().sum::<Float>() / samples.len() as Float;
        assert!((mean - 1.0 / lambda).abs() < 0.01, "Sample mean {} not close to expected {}", mean, 1.0 / lambda);
    }

    #[test]
    fn all_time_units() {
        let lambda = 1.0;
        let units = [
            TimeUnit::Days,
            TimeUnit::Hours,
            TimeUnit::Minutes,
            TimeUnit::Seconds,
            TimeUnit::Millis,
            TimeUnit::Nanos,
        ];

        for &unit in &units {
            let mut dist = Exponential::new(lambda, unit);
            let sample = dist.sample_at_t0().as_secs_float();
            assert!(sample >= 0.0, "Sample {} should be >= 0 for unit {:?}", sample, unit);
        }
    }
}

#[cfg(test)]
mod tests_tv {
    use super::*;
    use crate::time::{TimeUnit, DurationExtension};
    use std::time::Duration;

    #[test]
    fn samples_positive() {
        let mut dist = ExponentialTV::new(|t| 1.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);

        for i in 0..10 {
            let t = Duration::from_secs(i);
            let value = dist.sample(t).as_secs_float();
            assert!(value >= 0.0, "ExponentialTV sample should be >= 0, got {} at t={:?}", value, t);
        }
    }

    #[test]
    fn seeded_reproducible() {
        let seed = 123;
        let mut dist1 = ExponentialTV::new_seeded(|_| 1.5, TimeUnit::Seconds, seed);
        let mut dist2 = ExponentialTV::new_seeded(|_| 1.5, TimeUnit::Seconds, seed);

        for _ in 0..100 {
            let val1 = dist1.sample_at_t0().as_secs_float();
            let val2 = dist2.sample_at_t0().as_secs_float();
            assert_eq!(val1, val2, "Values {} and {} should be equal with the same seed", val1, val2);
        }
    }

    #[test]
    #[should_panic]
    fn invalid_lambda_panics() {
        let mut dist = ExponentialTV::new(|_| 0.0, TimeUnit::Seconds);
        dist.sample_at_t0();
    }

    #[test]
    #[ignore]
    fn mean_time_varying() {
        let mut dist = ExponentialTV::new(|t| 1.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);

        for t_sec in [0, 5, 10] {
            let t = Duration::from_secs(t_sec);
            let lambda = 1.0 + t_sec as Float * 0.1;
            let mut sum = 0.0;
            let n_samples = 100_000;

            for _ in 0..n_samples {
                sum += dist.sample(t).as_secs_float();
            }

            let mean = sum / n_samples as Float;
            assert!((mean - 1.0 / lambda).abs() < 0.02, "At t={}, mean {} not close to expected {}", t_sec, mean, 1.0 / lambda);
        }
    }

    #[test]
    fn all_time_units() {
        let units = [
            TimeUnit::Days,
            TimeUnit::Hours,
            TimeUnit::Minutes,
            TimeUnit::Seconds,
            TimeUnit::Millis,
            TimeUnit::Nanos,
        ];

        for &unit in &units {
            let mut dist_tv = ExponentialTV::new(|_| 1.0, unit);
            let sample_tv = dist_tv.sample(Duration::from_secs(0)).as_secs_float();
            assert!(sample_tv >= 0.0, "ExponentialTV sample {} should be >= 0 for unit {:?}", sample_tv, unit);
        }
    }
}
