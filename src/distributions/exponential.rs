//! Exponential distribution

use std::time::Duration;
use rand::{Rng, RngCore};

use crate::{
    time::TimeUnit,
    Float,
};
use super::Distribution;

/// Exponential distribution.
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, Exponential};
/// use opencct::TimeUnit;
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
    /// Time unit
    unit    : TimeUnit,
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
        Self { lambda, unit }
    }
}

impl Distribution for Exponential {
    fn sample(&self, _: Duration, rng: &mut dyn RngCore) -> Duration {
        let raw = -rng.random::<Float>().ln() / self.lambda;
        self.unit.to(raw)
    }

    fn mean(&self, _: Duration) -> Duration {
        self.unit.to(1.0 / self.lambda)
    }

    fn variance(&self, _: Duration) -> Duration {
        self.unit.to2(1.0 / self.lambda.powi(2))
    }
}

/// Exponential distribution with time-varying rate parameter.
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, ExponentialTV};
/// use opencct::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = ExponentialTV::new(
///     |t| 1.0 + TimeUnit::Seconds.from(t) * 0.1,
///     TimeUnit::Seconds,
/// );
/// let sample = dist.sample(Duration::from_secs(10), &mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct ExponentialTV<F> {
    /// Rate parameter as a function of time
    lambda   : F,
    /// Time unit
    unit     : TimeUnit,
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
        Self { lambda, unit }
    }

    /// Get the lambda parameter at a given point in time
    fn get_lambda_at(&self, at: Duration) -> Float {
        let lambda = (self.lambda)(at);
        debug_assert!(lambda > 0.0, "Invalid lambda at {at:?}: {lambda}");
        lambda
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
    fn sample(&self, at: Duration, rng: &mut dyn RngCore) -> Duration {
        let lambda = self.get_lambda_at(at);
        let raw = -rng.random::<Float>().ln() / lambda;
        self.unit.to(raw)
    }

    /// See [Distribution::mean]
    /// # Panic
    /// In debug, this function will panic if at the requested time the rate parameter <= 0.
    /// **This is NOT checked in release mode!**
    fn mean(&self, at: Duration) -> Duration {
        self.unit.to(1.0 / self.get_lambda_at(at))
    }

    /// See [Distribution::variance]
    /// # Panic
    /// In debug, this function will panic if at the requested time the rate parameter <= 0.
    /// **This is NOT checked in release mode!**
    fn variance(&self, at: Duration) -> Duration {
        self.unit.to2(1.0 / self.get_lambda_at(at).powi(2))
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
                let x = TimeUnit::Seconds.from(dist.sample_at_t0(&mut rng));
                assert!(x.is_finite(), "Generated value is not finite: {x}");
            }
        }

        #[test]
        #[ignore] // statistical test, expensive
        fn mean_and_variance_large_sample() {
            const N_SAMPLES: usize = 500_000;
            const MEAN_TOL: Float = 0.01;
            const VAR_TOL: Float = 0.02;

            let lambda = 2.0;
            let dist = Exponential::new(lambda, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            let samples = dist.sample_n_at_t0(N_SAMPLES, &mut rng);

            let stats = BasicStatistics::compute(&samples);

            assert_close(stats.mean(), dist.mean_at_t0(), MEAN_TOL, "Exponential mean");
            assert_close(stats.variance(), dist.variance_at_t0(), VAR_TOL, "Exponential variance");
        }
    }

    mod exponential_tv {
        use super::*;

        #[test]
        fn smoke_sample_tv() {
            let dist = ExponentialTV::new(
                |t| 1.0 + TimeUnit::Seconds.from(t) * 0.1,
                TimeUnit::Seconds,
            );
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
            const N_SAMPLES: usize = 500_000;
            const MEAN_TOL: Float = 0.02;
            const VAR_TOL: Float = 0.03;

            let dist = ExponentialTV::new(
                |t| 1.0 + TimeUnit::Seconds.from(t) * 0.1,
                TimeUnit::Seconds,
            );
            let mut rng = StdRng::from_os_rng();

            for t_sec in [0, 5, 10] {
                let t = Duration::from_secs(t_sec);

                let samples = dist.sample_n(N_SAMPLES, t, &mut rng);

                let stats = BasicStatistics::compute(&samples);

                assert_close(
                    stats.mean(),
                    dist.mean(t),
                    MEAN_TOL,
                    &format!("ExponentialTV mean at t={}", t_sec),
                );
                assert_close(
                    stats.variance(),
                    dist.variance(t),
                    VAR_TOL,
                    &format!("ExponentialTV variance at t={}", t_sec),
                );
            }
        }
    }
}
