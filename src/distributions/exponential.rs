//! Exponential distribution

use std::time::Duration;
use rand::Rng;

use crate::{
    time::{DurationExtension, TimeUnit},
    Float,
};
use super::Distribution;

/// Exponential distribution.
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, Exponential};
/// use opencct::time::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = Exponential::new(1.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0(&mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Exponential {
    /// The rate parameter (> 0)
    lambda  : Float,
    /// Time unit factor
    factor  : Float,
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
        Self { lambda, factor: unit.factor() }
    }
}

impl Distribution for Exponential {
    fn sample<R: Rng + ?Sized>(&self, _: Duration, rng: &mut R) -> Duration {
        let raw = -rng.random::<Float>().ln() / self.lambda;
        Duration::from_secs_float(raw * self.factor)
    }
}

/// Exponential distribution with time-varying rate parameter.
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, ExponentialTV};
/// use opencct::time::{DurationExtension, TimeUnit};
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = ExponentialTV::new(|t| 1.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
/// let sample = dist.sample(Duration::from_secs(10), &mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct ExponentialTV<F> {
    /// Rate parameter as a function of time
    lambda   : F,
    /// Time unit factor
    factor  : Float,
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
        Self { lambda, factor: unit.factor() }
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
    fn sample<R: Rng + ?Sized>(&self, at: Duration, rng: &mut R) -> Duration {
        let lambda = (self.lambda)(at);
        debug_assert!(lambda > 0.0, "Invalid lambda at {at:?}: {lambda}");
        let raw = -rng.random::<Float>().ln() / lambda;
        Duration::from_secs_float(raw * self.factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use crate::test_utils::{BasicStatistics, assert_close};

    mod exponential {
        use super::*;

        #[test]
        #[should_panic]
        fn new_panics_on_zero_lambda() {
            let _ = Exponential::new(0.0, TimeUnit::Seconds);
        }

        #[test]
        fn smoke_sample() {
            let dist = Exponential::new(1.5, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();
            let _ = dist.sample_at_t0(&mut rng);
        }

        #[test]
        fn values_are_finite() {
            let dist = Exponential::new(1.0, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            for _ in 0..10_000 {
                let x = dist.sample_at_t0(&mut rng).as_secs_float();
                assert!(x.is_finite(), "Generated value is not finite: {x}");
            }
        }

        #[test]
        #[ignore] // statistical test, expensive
        fn mean_and_variance_large_sample() {
            const N_SAMPLES: usize = 100_000;
            const MEAN_TOL: Float = 0.01;
            const VAR_TOL: Float = 0.02;

            let lambda = 2.0;
            let dist = Exponential::new(lambda, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            let samples: Vec<Float> = (0..N_SAMPLES)
                .map(|_| dist.sample_at_t0(&mut rng).as_secs_float())
                .collect();

            let stats = BasicStatistics::compute(&samples);
            let expected_mean = 1.0 / lambda;
            let expected_var = 1.0 / (lambda * lambda);

            assert_close(stats.mean(), expected_mean, MEAN_TOL, "Exponential mean");
            assert_close(stats.variance(), expected_var, VAR_TOL, "Exponential variance");
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
            let mut rng = StdRng::from_os_rng();

            for &unit in &units {
                let dist = Exponential::new(lambda, unit);
                let sample = dist.sample_at_t0(&mut rng).as_secs_float();
                assert!(sample >= 0.0, "Sample {} should be >= 0 for unit {:?}", sample, unit);
            }
        }
    }

    mod exponential_tv {
        use super::*;

        #[test]
        fn smoke_sample_tv() {
            let dist = ExponentialTV::new(|t| 1.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();
            let _ = dist.sample_at_t0(&mut rng);
            let _ = dist.sample(Duration::from_secs(5), &mut rng);
        }

        #[test]
        #[should_panic]
        fn invalid_lambda_panics() {
            let dist = ExponentialTV::new(|_| 0.0, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();
            dist.sample_at_t0(&mut rng);
        }

        #[test]
        #[ignore] // statistical test, expensive
        fn mean_and_variance_time_varying() {
            const N_SAMPLES: usize = 100_000;
            const MEAN_TOL: Float = 0.02;
            const VAR_TOL: Float = 0.03;

            let dist = ExponentialTV::new(|t| 1.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            for t_sec in [0, 5, 10] {
                let t = Duration::from_secs(t_sec);
                let lambda = 1.0 + t_sec as Float * 0.1;

                let samples: Vec<Float> = (0..N_SAMPLES)
                    .map(|_| dist.sample(t, &mut rng).as_secs_float())
                    .collect();

                let stats = BasicStatistics::compute(&samples);
                let expected_mean = 1.0 / lambda;
                let expected_var = 1.0 / (lambda * lambda);

                assert_close(
                    stats.mean(),
                    expected_mean,
                    MEAN_TOL,
                    &format!("ExponentialTV mean at t={}", t_sec),
                );
                assert_close(
                    stats.variance(),
                    expected_var,
                    VAR_TOL,
                    &format!("ExponentialTV variance at t={}", t_sec),
                );
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
            let mut rng = StdRng::from_os_rng();

            for &unit in &units {
                let dist_tv = ExponentialTV::new(|_| 1.0, unit);
                let sample_tv = dist_tv.sample_at_t0(&mut rng).as_secs_float();
                assert!(sample_tv >= 0.0, "ExponentialTV sample {} should be >= 0 for unit {:?}", sample_tv, unit);
            }
        }
    }
}
