//! Log-normal distribution

use std::time::Duration;
use rand::{
    rngs::StdRng,
    SeedableRng,
};

use crate::{
    time::{DurationExtension, TimeUnit},
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
/// use opencct::distributions::{Distribution, LogNormal};
/// use opencct::time::TimeUnit;
///
/// let mut dist = LogNormal::new(5.0, 1.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct LogNormal {
    /// Logarithm of location
    mu      : Float,
    /// Logarithm of scale (> 0)
    sigma   : Float,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
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
        Self { mu, sigma, factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    /// Create a new [LogNormal] distribution with a specified random seed.
    /// # Arguments
    /// * `mu` - Mu parameter, logarithm of location
    /// * `sigma` - Sigma parameter, logarithm of scale
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// A new [LogNormal].
    /// # Panic
    /// This function panics if `sigma <= 0`
    pub fn new_seeded(mu: Float, sigma: Float, unit: TimeUnit, seed: u64) -> Self {
        assert!(sigma > 0.0, "Sigma ({sigma}) must be > 0");
        Self { mu, sigma, factor: unit.factor(), rng: StdRng::seed_from_u64(seed) }
    }
}

impl Distribution for LogNormal {
    fn sample(&mut self, _: Duration) -> Duration {
        let x = scaled_zignor_method(
            &mut self.rng,
            self.mu,
            self.sigma,
        );
        Duration::from_secs_float(x.exp() * self.factor)
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
/// use opencct::distributions::{Distribution, LogNormalTV};
/// use opencct::time::{TimeUnit, DurationExtension};
///
/// let mut dist = LogNormalTV::new(|t| 1.0 + t.as_secs_float() * 0.1, |t| 3.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct LogNormalTV<FMu, FSigma> {
    /// The logarithm of location function of time
    mu      : FMu,
    /// The logarithm of scale as a function of time
    sigma   : FSigma,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
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
        Self { mu, sigma, factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    /// Create a new [LogNormalTV] distribution with a specified random seed.
    /// # Arguments
    /// * `mu` - Function to compute the logarithm of location at a given time.
    /// * `sigma` - Function to compute the logarithm of scale at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// A new [LogNormalTV].
    /// # Be careful!
    /// `sigma` bound is not checked in release mode! Make sure you fulfill it!
    pub fn new_seeded(mu: FMu, sigma: FSigma, unit: TimeUnit, seed: u64) -> Self {
        Self { mu, sigma, factor: unit.factor(), rng: StdRng::seed_from_u64(seed) }
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
    fn sample(&mut self, at: Duration) -> Duration {
        let (mu, sigma) = ((self.mu)(at), (self.sigma)(at));
        debug_assert!(sigma > 0.0, "Invalid sigma at {at:?}: {sigma}");
        let x = scaled_zignor_method(
            &mut self.rng,
            mu,
            sigma,
        );
        Duration::from_secs_float(x.exp() * self.factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{BasicStatistics, assert_close};

    #[test]
    fn samples_positive() {
        let mut dist = LogNormal::new(0.0, 1.0, TimeUnit::Seconds);

        for _ in 0..100 {
            let sample = dist.sample_at_t0();
            assert!(
                sample.as_secs_float() > 0.0,
                "LogNormal sample should be > 0, got {sample:?}",
            );
        }
    }

    #[test]
    fn seeded_reproducible() {
        let mu = 0.0;
        let sigma = 1.0;
        let seed = 42;
        let mut dist1 = LogNormal::new_seeded(mu, sigma, TimeUnit::Seconds, seed);
        let mut dist2 = LogNormal::new_seeded(mu, sigma, TimeUnit::Seconds, seed);

        for _ in 0..50 {
            let val1 = dist1.sample_at_t0();
            let val2 = dist2.sample_at_t0();
            assert_eq!(
                val1, val2,
                "Values {val1:?} and {val2:?} should be equal with the same seed",
            );
        }
    }

    #[test]
    #[should_panic]
    fn invalid_sigma_panics() {
        let _ = LogNormal::new(0.0, 0.0, TimeUnit::Seconds);
    }

    #[test]
    #[ignore]
    fn mean_and_variance() {
        let mu: Float = 1.0;
        let sigma: Float = 0.5;
        const N_SAMPLES: usize = 100_000;

        let mut dist = LogNormal::new(mu, sigma, TimeUnit::Seconds);

        let samples: Vec<Float> = (0..N_SAMPLES)
            .map(|_| dist.sample_at_t0().as_secs_float())
            .collect();

        let stats = BasicStatistics::compute(&samples);
        let (mean, var) = (stats.mean(), stats.variance());

        let expected_mean = (mu + 0.5 * sigma.powi(2)).exp();
        let expected_var = ((sigma.powi(2)).exp() - 1.0) * (2.0 * mu + sigma.powi(2)).exp();

        assert_close(mean, expected_mean, 0.05, "LogNormal mean"); // 5% tolerance
        assert_close(var, expected_var, 0.10, "LogNormal variance"); // 10% tolerance
    }
}

#[cfg(test)]
mod tests_tv {
    use super::*;
    use crate::test_utils::{BasicStatistics, assert_close};

    #[test]
    fn samples_positive() {
        let mut dist = LogNormalTV::new(|_| 0.0, |_| 1.0, TimeUnit::Seconds);

        for i in 0..10 {
            let t = Duration::from_secs(i);
            let sample = dist.sample(t);
            assert!(
                sample.as_secs_float() > 0.0,
                "LogNormalTV sample should be > 0, got {sample:?} at t={t:?}",
            );
        }
    }

    #[test]
    fn seeded_reproducible() {
        let seed = 123;
        let mut dist1 = LogNormalTV::new_seeded(|_| 0.0, |_| 1.0, TimeUnit::Seconds, seed);
        let mut dist2 = LogNormalTV::new_seeded(|_| 0.0, |_| 1.0, TimeUnit::Seconds, seed);

        for _ in 0..50 {
            let val1 = dist1.sample_at_t0();
            let val2 = dist2.sample_at_t0();
            assert_eq!(
                val1, val2,
                "Values {val1:?} and {val2:?} should be equal with the same seed",
            );
        }
    }

    #[test]
    #[ignore]
    fn mean_and_variance_time_varying() {
        let mu_fn = |t: Duration| 0.5 * t.as_secs_float();
        let sigma_fn = |_| 0.25;
        const N_SAMPLES: usize = 100_000;

        let mut dist = LogNormalTV::new(mu_fn, sigma_fn, TimeUnit::Seconds);

        for &t_sec in &[0, 4, 8] {
            let t = Duration::from_secs(t_sec);
            let mu = mu_fn(t);
            let sigma = sigma_fn(t);

            let samples: Vec<Float> = (0..N_SAMPLES)
                .map(|_| dist.sample(t).as_secs_float())
                .collect();

            let stats = BasicStatistics::compute(&samples);
            let (mean, var) = (stats.mean(), stats.variance());
            let expected_mean = (mu + 0.5 * sigma.powi(2)).exp();
            let expected_var = ((sigma.powi(2)).exp() - 1.0) * (2.0 * mu + sigma.powi(2)).exp();

            assert_close(mean, expected_mean, 0.05, &format!("LogNormalTV mean at t={t_sec}"));
            assert_close(var, expected_var, 0.10, &format!("LogNormalTV variance at t={t_sec}"));
        }
    }
}
