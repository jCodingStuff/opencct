//! Log-normal distribution

use std::time::Duration;
use rand::RngCore;

use crate::{
    time::TimeUnit,
    Float,
};
use super::{
    Distribution,
    algorithms::zignor::scaled_zignor_method,
};

/// Log-normal distribution.
///
/// Implemented via the ZIGNOR variant of the Ziggurat method. See
/// Doornik, J. A. (2005).
/// [An Improved Ziggurat Method to Generate Normal Random Samples](https://www.doornik.com/research/ziggurat.pdf).
/// University of Oxford.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, LogNormal};
/// use opencct::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = LogNormal::new(5.0, 1.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0(&mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct LogNormal {
    /// Logarithm of location
    mu      : Float,
    /// Logarithm of scale (> 0)
    sigma   : Float,
    /// Time unit
    unit    : TimeUnit,
}

impl LogNormal {
    /// Create a new [LogNormal] distribution with given mu and sigma.
    /// # Arguments
    /// * `mu` - Mu parameter, logarithm of location
    /// * `sigma` - Sigma parameter, logarithm of scale
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [LogNormal].
    /// # Panic
    /// This function panics if `sigma <= 0`
    pub fn new(mu: Float, sigma: Float, unit: TimeUnit) -> Self {
        assert!(sigma > 0.0, "Sigma ({sigma}) must be > 0");
        Self { mu, sigma, unit }
    }
}

impl Distribution for LogNormal {
    fn sample(&self, _: Duration, rng: &mut dyn RngCore) -> Duration {
        let x = scaled_zignor_method(rng, self.mu, self.sigma);
        self.unit.to(x.exp())
    }

    fn mean(&self, _: Duration) -> Duration {
        let raw = (self.mu + 0.5 * self.sigma.powi(2)).exp();
        self.unit.to(raw)
    }

    fn variance(&self, _: Duration) -> Duration {
        let raw = self.sigma.powi(2).exp_m1() * (2.0 * self.mu + self.sigma.powi(2)).exp();
        self.unit.to2(raw)
    }
}

/// Log-normal distribution with time-varying parmeters.
///
/// Implemented via the ZIGNOR variant of the Ziggurat method. See
/// Doornik, J. A. (2005).
/// [An Improved Ziggurat Method to Generate Normal Random Samples](https://www.doornik.com/research/ziggurat.pdf).
/// University of Oxford.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, LogNormalTV};
/// use opencct::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = LogNormalTV::new(
///     |t| 1.0 + TimeUnit::Seconds.from(t) * 0.1,
///     |t| 3.0 + TimeUnit::Seconds.from(t) * 0.1,
///     TimeUnit::Seconds,
/// );
/// let sample = dist.sample(Duration::from_secs(10), &mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct LogNormalTV<FMu, FSigma> {
    /// The logarithm of location function of time
    mu      : FMu,
    /// The logarithm of scale as a function of time
    sigma   : FSigma,
    /// Time unit
    unit    : TimeUnit,
}

impl<FMu, FSigma> LogNormalTV<FMu, FSigma>
where
    FMu     : Fn(Duration) -> Float,
    FSigma  : Fn(Duration) -> Float,
{
    /// Create a new [LogNormalTV] distribution with given parameter functions.
    /// # Arguments
    /// * `mu` - Function to compute the logarithm of location at a given time.
    /// * `sigma` - Function to compute the logarithm of scale at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// A new [LogNormalTV].
    /// # Be careful!
    /// `sigma` bound is not checked in release mode! Make sure you fulfill it!
    pub fn new(mu: FMu, sigma: FSigma, unit: TimeUnit) -> Self {
        Self { mu, sigma, unit }
    }

    /// Get the parameters (mu, sigma) of the distribution at a given point in time
    fn get_parameters_at(&self, at: Duration) -> (Float, Float) {
        let (mu, sigma) = ((self.mu)(at), (self.sigma)(at));
        debug_assert!(sigma > 0.0, "Invalid sigma at {at:?}: {sigma}");
        (mu, sigma)
    }
}

impl<FMu, FSigma> Distribution for LogNormalTV<FMu, FSigma>
where
    FMu: Fn(Duration) -> Float,
    FSigma: Fn(Duration) -> Float,
{
    /// See [Distribution::sample]
    /// # Panic
    /// In debug, this function will panic if at the requested time the logarithm of scale <= 0
    /// **This is NOT checked in release mode!**
    fn sample(&self, at: Duration, rng: &mut dyn RngCore) -> Duration {
        let (mu, sigma) = self.get_parameters_at(at);
        let x = scaled_zignor_method(rng, mu, sigma);
        self.unit.to(x.exp())
    }

    /// See [Distribution::mean]
    /// # Panic
    /// In debug, this function will panic if at the requested time the logarithm of scale <= 0
    /// **This is NOT checked in release mode!**
    fn mean(&self, at: Duration) -> Duration {
        let (mu, sigma) = self.get_parameters_at(at);
        let raw = (mu + 0.5 * sigma.powi(2)).exp();
        self.unit.to(raw)
    }

    /// See [Distribution::variance]
    /// # Panic
    /// In debug, this function will panic if at the requested time the logarithm of scale <= 0
    /// **This is NOT checked in release mode!**
    fn variance(&self, at: Duration) -> Duration {
        let (mu, sigma) = self.get_parameters_at(at);
        let raw = sigma.powi(2).exp_m1() * (2.0 * mu + sigma.powi(2)).exp();
        self.unit.to2(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use crate::test_utils::{BasicStatistics, assert_close};

    mod lognormal {
        use super::*;

        #[test]
        fn samples_positive() {
            let dist = LogNormal::new(0.0, 1.0, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            for _ in 0..100 {
                let sample = dist.sample_at_t0(&mut rng);
                assert!(
                    sample > Duration::ZERO,
                    "LogNormal sample should be > 0, got {sample:?}",
                );
            }
        }

        #[test]
        #[should_panic]
        fn invalid_sigma_panics() {
            let _ = LogNormal::new(0.0, 0.0, TimeUnit::Seconds);
        }

        #[test]
        #[ignore] // statistical test, expensive
        fn mean_and_variance() {
            let mu: Float = 1.0;
            let sigma: Float = 0.5;
            const N_SAMPLES: usize = 500_000;

            let dist = LogNormal::new(mu, sigma, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            let samples: Vec<_> = dist.sample_n_at_t0(N_SAMPLES, &mut rng);

            let stats = BasicStatistics::compute(&samples);

            assert_close(stats.mean(), dist.mean_at_t0(), 0.05, "LogNormal mean"); // 5% tolerance
            assert_close(stats.variance(), dist.variance_at_t0(), 0.10, "LogNormal variance"); // 10% tolerance
        }
    }

    mod lognormal_tv {
        use super::*;

        #[test]
        fn samples_positive() {
            let dist = LogNormalTV::new(|_| 0.0, |_| 1.0, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            for i in 0..10 {
                let t = Duration::from_secs(i);
                let sample = dist.sample(t, &mut rng);
                assert!(
                    sample > Duration::ZERO,
                    "LogNormalTV sample should be > 0, got {sample:?} at t={t:?}",
                );
            }
        }

        #[test]
        #[ignore]
        fn mean_and_variance_time_varying() {
            const N_SAMPLES: usize = 500_000;

            let dist = LogNormalTV::new(
                |t: Duration| 0.5 * TimeUnit::Seconds.from(t) + 0.1,
                |_| 0.25,
                TimeUnit::Seconds,
            );
            let mut rng = StdRng::from_os_rng();

            for t_sec in [0, 4, 8] {
                let t = Duration::from_secs(t_sec);
                let samples = dist.sample_n(N_SAMPLES, t, &mut rng);

                let stats = BasicStatistics::compute(&samples);

                assert_close(
                    stats.mean(),
                    dist.mean(t),
                    0.05,
                    &format!("LogNormalTV mean at t={t_sec}"),
                );
                assert_close(
                    stats.variance(),
                    dist.variance(t),
                    0.10,
                    &format!("LogNormalTV variance at t={t_sec}"),
                );
            }
        }
    }
}
