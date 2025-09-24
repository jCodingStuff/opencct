//! Normal distribution

use std::time::Duration;
use rand::RngCore;

use crate::{
    time::{DurationExtension, TimeUnit},
    Float,
};
use super::{
    Distribution,
    algorithms::zignor::scaled_zignor_method,
};

/// Normal distribution. Since in the current context, negative time does not make sense, the negative values
/// will be clamped to 0.
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
/// use opencct::distributions::{Distribution, Normal};
/// use opencct::time::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = Normal::new(5.0, 1.0, TimeUnit::Millis);
/// let sample = dist.sample_at_t0(&mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Normal {
    /// The mean (>= 0)
    mu      : Float,
    /// The standard deviation (> 0)
    sigma   : Float,
    /// Time unit factor
    factor  : Float,
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
        Self { mu, sigma, factor: unit.factor() }
    }

    /// Get the theoretical mean of the distribution
    pub fn mean(&self) -> Float { self.mu }

    /// Get the theoretical variance of the distribution
    pub fn variance(&self) -> Float { self.sigma.powi(2) }
}

impl Distribution for Normal {
    fn sample(&self, _: Duration, rng: &mut dyn RngCore) -> Duration {
        let x = scaled_zignor_method(rng, self.mu, self.sigma);
        Duration::from_secs_float(x.max(0.0) * self.factor)
    }
}

/// Normal distribution with time-varying parmeters. Since in the current context,
/// negative time does not make sense, the negative values will be clamped to 0.
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
/// use opencct::distributions::{Distribution, NormalTV};
/// use opencct::time::{TimeUnit, DurationExtension};
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = NormalTV::new(|t| 1.0 + t.as_secs_float() * 0.1, |t| 3.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
/// let sample = dist.sample(Duration::from_secs(10), &mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct NormalTV<FMu, FSigma> {
    /// The mean as a function of time
    mu      : FMu,
    /// The standard deviation as a function of time
    sigma   : FSigma,
    /// Time unit factor
    factor  : Float,
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
        Self { mu, sigma, factor: unit.factor() }
    }

    /// Get the parameters (mu, sigma) of the distribution at a given point in time
    fn get_parameters_at(&self, at: Duration) -> (Float, Float) {
        let (mu, sigma) = ((self.mu)(at), (self.sigma)(at));
        debug_assert!(sigma > 0.0, "Invalid sigma at {at:?}: {sigma}");
        (mu, sigma)
    }

    /// Get the theoretical mean of the distribution at a given time point
    pub fn mean_at(&self, at: Duration) -> Float { self.get_parameters_at(at).0 }

    /// Get the theoretical variance of the distribution at a given time point
    pub fn variance_at(&self, at: Duration) -> Float { self.get_parameters_at(at).1.powi(2) }
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
    fn sample(&self, at: Duration, rng: &mut dyn RngCore) -> Duration {
        let (mu, sigma) = self.get_parameters_at(at);
        let x = scaled_zignor_method(rng, mu, sigma);
        Duration::from_secs_float(x.max(0.0) * self.factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use crate::test_utils::{BasicStatistics, assert_close};

    mod normal {
        use super::*;

        #[test]
        fn samples_positive() {
            let dist = Normal::new(5.0, 2.0, TimeUnit::Millis);
            let mut rng = StdRng::from_os_rng();

            for _ in 0..100 {
                let sample = dist.sample_at_t0(&mut rng).as_millis_float();
                assert!(sample >= 0.0, "Normal sample should be >= 0, got {sample}");
            }
        }

        #[test]
        #[should_panic]
        fn invalid_sigma_panics() {
            let _ = Normal::new(5.0, 0.0, TimeUnit::Millis);
        }

        #[test]
        #[ignore]
        fn mean_and_variance() {
            const N_SAMPLES: usize = 500_000;
            let mu = 50.0;
            let sigma = 2.0;

            let dist = Normal::new(mu, sigma, TimeUnit::Millis);
            let mut rng = StdRng::from_os_rng();
            let samples: Vec<_> = dist.sample_n_at_t0(N_SAMPLES, &mut rng)
                .iter()
                .map(|d| d.as_millis_float())
                .collect();

            let stats = BasicStatistics::compute(&samples);

            assert_close(stats.mean(), dist.mean(), 0.01, "Normal mean");      // 1% tolerance
            assert_close(stats.variance(), dist.variance(), 0.02, "Normal variance"); // 2% tolerance
        }
    }

    mod normal_tv {
        use super::*;

        #[test]
        fn samples_positive() {
            let dist = NormalTV::new(
                |t| 5.0 + t.as_secs_float() * 0.1,
                |t| 2.0 + t.as_secs_float() * 0.05,
                TimeUnit::Millis,
            );
            let mut rng = StdRng::from_os_rng();

            for i in 0..10 {
                let t = Duration::from_secs(i);
                let sample = dist.sample(t, &mut rng).as_millis_float();
                assert!(sample >= 0.0, "NormalTV sample should be >= 0, got {sample} at t={t:?}");
            }
        }

        #[test]
        #[ignore]
        fn mean_and_variance_time_varying() {
            const N_SAMPLES: usize = 500_000;

            let dist = NormalTV::new(
                |t| 50.0 + t.as_secs_float() * 0.1,
                |t| 2.0 + t.as_secs_float() * 0.05,
                TimeUnit::Millis,
            );
            let mut rng = StdRng::from_os_rng();

            for t_sec in [0, 5, 10] {
                let t = Duration::from_secs(t_sec);
                let samples: Vec<Float> = dist.sample_n(N_SAMPLES, t, &mut rng)
                    .iter()
                    .map(|d| d.as_millis_float())
                    .collect();

                let stats = BasicStatistics::compute(&samples);

                assert_close(stats.mean(), dist.mean_at(t), 0.01, &format!("NormalTV mean at t={t_sec}"));
                assert_close(stats.variance(), dist.variance_at(t), 0.02, &format!("NormalTV variance at t={t_sec}"));
            }
        }
    }
}
