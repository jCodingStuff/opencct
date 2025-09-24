//! Weibull distribution

use std::time::Duration;
use rand::{Rng, RngCore};

use crate::{
    time::TimeUnit,
    Float,
    math::gamma,
};
use super::Distribution;

/// Weibull distribution.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, Weibull};
/// use opencct::time::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = Weibull::new(1.0, 3.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0(&mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Weibull {
    /// Scale parameter
    lambda  : Float,
    /// Shape parameter
    k       : Float,
    /// Time unit
    unit    : TimeUnit,
}

impl Weibull {
    /// Create a new [Weibull] distribution with given shape and scale values.
    /// # Arguments
    /// * `lambda` - Scale
    /// * `k` - Shape
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [Weibull].
    /// # Panic
    /// This function panics if either `lambda` or `k` are <= 0
    pub fn new(lambda: Float, k: Float, unit: TimeUnit) -> Self {
        assert!(lambda > 0.0 && k > 0.0, "Invalid paramters: [lambda: {lambda}, k: {k}]");
        Self { lambda, k, unit }
    }
}

impl Distribution for Weibull {
    fn sample(&self, _: Duration, rng: &mut dyn RngCore) -> Duration {
        let raw = self.lambda * (-rng.random::<Float>().ln()).powf(1.0 / self.k);
        self.unit.to_duration(raw)
    }

    fn mean(&self, _: Duration) -> Duration {
        let raw = self.lambda * gamma(1.0 + 1.0 / self.k);
        self.unit.to_duration(raw)
    }

    fn variance(&self, _: Duration) -> Duration {
        let raw = self.lambda.powi(2) * (gamma(1.0 + 2.0/self.k) - gamma(1.0 + 1.0/self.k).powi(2));
        self.unit.to_duration(raw)
    }
}

/// Weibull distribution with time-varying shape and scale.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, WeibullTV};
/// use opencct::time::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = WeibullTV::new(
///     |t| 1.0 + TimeUnit::Seconds.from_duration(t) * 0.1,
///     |t| 3.0 + TimeUnit::Seconds.from_duration(t) * 0.1,
///     TimeUnit::Seconds,
/// );
/// let sample = dist.sample(Duration::from_secs(10), &mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct WeibullTV<Fl, Fk> {
    /// Scale parameter as a function of time
    lambda  : Fl,
    /// Shape parameter as a function of time
    k       : Fk,
    /// Time unit
    unit    : TimeUnit,
}

impl<Fl, Fk> WeibullTV<Fl, Fk>
where
    Fl: Fn(Duration) -> Float,
    Fk: Fn(Duration) -> Float,
{
    /// Create a new [WeibullTV] distribution with given shape and scale functions.
    /// # Arguments
    /// * `lambda` - Function to compute the scale at a given time. Must be > 0 for any t >= 0
    /// * `k` - Function to compute the shape at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [WeibullTV].
    /// # Be careful!
    /// `lambda` and `k` values are not checked in release mode! Make sure you fulfill the contract!
    pub fn new(lambda: Fl, k: Fk, unit: TimeUnit) -> Self {
        Self { lambda, k, unit }
    }

    /// Get the parameters (lambda, k) of the distribution at a given point in time
    fn get_parameters_at(&self, at: Duration) -> (Float, Float) {
        let (lambda, k) = ((self.lambda)(at), (self.k)(at));
        debug_assert!(lambda > 0.0 && k > 0.0, "Invalid lambda {lambda} or k {k} bound at {at:?}");
        (lambda, k)
    }
}

impl<Fl, Fk> Distribution for WeibullTV<Fl, Fk>
where
    Fl: Fn(Duration) -> Float,
    Fk: Fn(Duration) -> Float,
{
    /// See [Distribution::sample]
    /// # Panic
    /// In debug, this function will panic if at the requested time the shape or scale are <= 0.
    /// **This is NOT checked in release mode!**
    fn sample(&self, at: Duration, rng: &mut dyn RngCore) -> Duration {
        let (lambda, k) = self.get_parameters_at(at);
        let raw = lambda * (-rng.random::<Float>().ln()).powf(1.0 / k);
        self.unit.to_duration(raw)
    }

    /// See [Distribution::mean]
    /// # Panic
    /// In debug, this function will panic if at the requested time the shape or scale are <= 0.
    /// **This is NOT checked in release mode!**
    fn mean(&self, at: Duration) -> Duration {
        let (lambda, k) = self.get_parameters_at(at);
        let raw = lambda * gamma(1.0 + 1.0 / k);
        self.unit.to_duration(raw)
    }

    /// See [Distribution::variance]
    /// # Panic
    /// In debug, this function will panic if at the requested time the shape or scale are <= 0.
    /// **This is NOT checked in release mode!**
    fn variance(&self, at: Duration) -> Duration {
        let (lambda, k) = self.get_parameters_at(at);
        let raw = lambda.powi(2) * (gamma(1.0 + 2.0/k) - gamma(1.0 + 1.0/k).powi(2));
        self.unit.to_duration(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use crate::test_utils::{assert_close, BasicStatistics};

    mod weibull {

        use super::*;

        #[test]
        fn smoke_test() {
            let dist = Weibull::new(2.0, 1.5, TimeUnit::Seconds);
            let mut rng = StdRng::seed_from_u64(42);
            let sample = dist.sample_at_t0(&mut rng);
            assert!(sample >= Duration::ZERO);
        }

        #[test]
        #[ignore]
        fn mean_and_variance() {
            const N_SAMPLES: usize = 500_000;
            let lambda: Float = 2.0;
            let k: Float = 1.5;
            let dist = Weibull::new(lambda, k, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            let samples = dist.sample_n_at_t0(N_SAMPLES, &mut rng);

            let stats = BasicStatistics::compute(&samples);

            assert_close(stats.mean(), dist.mean_at_t0(), 0.05, "Weibull mean");
            assert_close(stats.variance(), dist.variance_at_t0(), 0.10, "Weibull variance");
        }

        #[test]
        #[should_panic]
        fn invalid_params_zero() {
            Weibull::new(0.0, 1.0, TimeUnit::Seconds);
        }

        #[test]
        #[should_panic]
        fn invalid_params_negative() {
            Weibull::new(1.0, -1.0, TimeUnit::Seconds);
        }
    }

    mod weibull_tv {
        use super::*;

        #[test]
        fn smoke_test() {
            let dist = WeibullTV::new(|_| 2.0, |_| 1.5, TimeUnit::Seconds);
            let mut rng = StdRng::seed_from_u64(42);
            let sample = dist.sample_at_t0(&mut rng);
            assert!(sample >= Duration::ZERO);
        }

        #[test]
        #[ignore] // statistical test, expensive
        fn mean_and_variance_large_sample_tv() {
            const N_SAMPLES: usize = 500_000;

            let dist = WeibullTV::new(
                |t| 0.5 * TimeUnit::Seconds.from_duration(t) + 0.1,
                |t| 0.2 * TimeUnit::Seconds.from_duration(t) + 1.0,
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
                    0.01,
                    &format!("WeibullTV mean at t={t_sec}"),
                );
                assert_close(
                    stats.variance(),
                    dist.variance(t),
                    0.02,
                    &format!("WeibullTV variance at t={t_sec}"),
                );
            }
        }
    }
}
