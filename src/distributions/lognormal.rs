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
    use crate::time::DurationExtension;

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
    fn mean_matches_theory() {
        let mu: Float = 1.0;
        let sigma :Float = 0.5;
        let expected_mean = (mu + 0.5 * sigma * sigma).exp();

        let mut dist = LogNormal::new(mu, sigma, TimeUnit::Seconds);
        let n_samples = 100_000;
        let mut sum = 0.0;

        for _ in 0..n_samples {
            sum += dist.sample_at_t0().as_secs_float();
        }

        let empirical_mean = sum / n_samples as Float;

        assert!(
            (empirical_mean - expected_mean).abs() / expected_mean < 0.05,
            "Empirical mean {} not close to theoretical {}",
            empirical_mean,
            expected_mean
        );
    }
}

#[cfg(test)]
mod tests_tv {
    use super::*;

    #[test]
    fn samples_positive() {
        let mut dist = LogNormalTV::new(
            |_| 0.0,
            |_| 1.0,
            TimeUnit::Seconds,
        );

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
    fn mean_increases_with_mu() {
        let mut dist = LogNormalTV::new(
            |t| t.as_secs_float() * 0.1,
            |_| 0.5,
            TimeUnit::Seconds,
        );

        let t1 = Duration::from_secs(1);
        let t2 = Duration::from_secs(5);

        let mut sum1 = 0.0;
        let mut sum2 = 0.0;
        let n_samples = 10_000;

        for _ in 0..n_samples {
            sum1 += dist.sample(t1).as_secs_float();
            sum2 += dist.sample(t2).as_secs_float();
        }

        let mean1 = sum1 / n_samples as Float;
        let mean2 = sum2 / n_samples as Float;

        assert!(
            mean2 > mean1,
            "Expected mean at t2={} ({}) to be greater than mean at t1={} ({})",
            t2.as_secs(),
            mean2,
            t1.as_secs(),
            mean1,
        );
    }

    #[test]
    #[ignore]
    fn mean_matches_theory_time_varying() {
        let mu_fn = |t: Duration| 0.5 * t.as_secs_float();
        let sigma_fn = |_| 0.25;
        let mut dist = LogNormalTV::new(mu_fn, sigma_fn, TimeUnit::Seconds);

        let t = Duration::from_secs(4);
        let mu = mu_fn(t);
        let sigma = sigma_fn(t);
        let expected_mean = (mu + 0.5 * sigma * sigma).exp();

        let n_samples = 100_000;
        let mut sum = 0.0;

        for _ in 0..n_samples {
            sum += dist.sample(t).as_secs_float();
        }

        let empirical_mean = sum / n_samples as Float;

        assert!(
            (empirical_mean - expected_mean).abs() / expected_mean < 0.05,
            "At t={t:?}, empirical mean {empirical_mean} not close to theoretical {expected_mean}",
        );
    }
}
