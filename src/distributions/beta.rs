//! Beta distribution

use std::time::Duration;
use rand::RngCore;

use crate::{
    time::{DurationExtension, TimeUnit},
    Float,
};
use super::{
    Distribution,
    algorithms::marsaglia_tsang::MarsagliaTsang,
};

/// Beta distribution.
///
/// Sampling comes from Gamma variables gnerated via the Marsaglia-Tsang method. See
/// Marsaglia, G., & Tsang, W. W. (2000).
/// [A simple method for generating gamma variables](https://doi.acm.org/10.1145/358407.358414).
/// *ACM Transactions on Mathematical Software (TOMS)*, 26(3), 363-372.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, Beta};
/// use opencct::time::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = Beta::new(1.0, 3.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0(&mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Beta {
    /// Time unit factor
    factor: Float,
    /// Sampling method struct for alpha
    method_alpha: MarsagliaTsang,
    /// Sampling method struct for beta
    method_beta: MarsagliaTsang,
}

impl Beta {
    /// Create a new [Beta] distribution with given shape parameters.
    /// # Arguments
    /// * `alpha` - Shape
    /// * `beta` - Scale
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [Beta].
    /// # Panic
    /// This function panics if either `alpha` or `beta` are <= 0
    pub fn new(alpha: Float, beta: Float, unit: TimeUnit) -> Self {
        assert!(alpha > 0.0 && beta > 0.0, "Invalid alpha {alpha} or beta {beta}");
        Self {
            factor          : unit.factor(),
            method_alpha    : MarsagliaTsang::setup(alpha),
            method_beta     : MarsagliaTsang::setup(beta),
        }
    }
}

impl Distribution for Beta {
    fn sample(&self, _: Duration, rng: &mut dyn RngCore) -> Duration {
        let x = self.method_alpha.sample_from_setup(rng, 1.0);
        let y = self.method_beta.sample_from_setup(rng, 1.0);
        Duration::from_secs_float(x / (x + y) * self.factor)
    }
}

/// Beta distribution with time-varying shape parameters.
///
/// Sampling comes from Gamma variables gnerated via the Marsaglia-Tsang method. See
/// Marsaglia, G., & Tsang, W. W. (2000).
/// [A simple method for generating gamma variables](https://doi.acm.org/10.1145/358407.358414).
/// *ACM Transactions on Mathematical Software (TOMS)*, 26(3), 363-372.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, BetaTV};
/// use opencct::time::{TimeUnit, DurationExtension};
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = BetaTV::new(|t| 1.0 + t.as_secs_float() * 0.1, |t| 3.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
/// let sample = dist.sample(Duration::from_secs(10), &mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct BetaTV<Fa, Fb> {
    /// Shape parameter
    alpha: Fa,
    /// Shape parameter
    beta: Fb,
    /// Time unit factor
    factor: Float,
}

impl<Fa, Fb> BetaTV<Fa, Fb>
where
    Fa: Fn(Duration) -> Float,
    Fb: Fn(Duration) -> Float,
{
    /// Create a new [BetaTV] distribution with given shape functions.
    /// # Arguments
    /// * `alpha` - Function to compute the shape at a given time. Must be > 0 for any t >= 0
    /// * `beta` - Function to compute the shape at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [BetaTV].
    /// # Be careful!
    /// `alpha` and `beta` values are not checked in release mode! Make sure you fulfill the contract!
    pub fn new(alpha: Fa, beta: Fb, unit: TimeUnit) -> Self {
        Self { alpha, beta, factor: unit.factor() }
    }
}

impl<Fa, Fb> Distribution for BetaTV<Fa, Fb>
where
    Fa: Fn(Duration) -> Float,
    Fb: Fn(Duration) -> Float,
{
    /// See [Distribution::sample]
    /// # Panic
    /// In debug, this function will panic if at the requested time either of the shape parameters is <= 0.
    /// **This is NOT checked in release mode!**
    fn sample(&self, at: Duration, rng: &mut dyn RngCore) -> Duration {
        let alpha = (self.alpha)(at);
        let beta = (self.beta)(at);
        debug_assert!(alpha > 0.0 && beta > 0.0, "Invalid alpha {alpha} or beta {beta} bound at {at:?}");
        let x = MarsagliaTsang::sample(rng, alpha, 1.0);
        let y = MarsagliaTsang::sample(rng, beta, 1.0);
        Duration::from_secs_float(x / (x + y) * self.factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use crate::test_utils::{BasicStatistics, assert_close};

    mod beta {
        use super::*;

        #[test]
        fn samples_in_bounds() {
            let dist = Beta::new(2.0, 5.0, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();
            for _ in 0..100 {
                let val = dist.sample_at_t0(&mut rng).as_secs_float();
                assert!(
                    val >= 0.0 && val <= 1.0,
                    "Sample {val} should be in [0,1]"
                );
            }
        }

        #[test]
        #[should_panic]
        fn invalid_alpha_panics() {
            let _ = Beta::new(0.0, 1.0, TimeUnit::Seconds);
        }

        #[test]
        #[should_panic]
        fn invalid_beta_panics() {
            let _ = Beta::new(1.0, -1.0, TimeUnit::Seconds);
        }

        #[test]
        #[ignore] // statistical test, expensive
        fn mean_and_variance() {
            let alpha = 2.0;
            let beta = 5.0;
            let dist = Beta::new(alpha, beta, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();
            const N: usize = 500_000;

            let samples: Vec<Float> = (0..N)
                .map(|_| dist.sample_at_t0(&mut rng).as_secs_float())
                .collect();

            let stats = BasicStatistics::compute(&samples);
            let expected_mean = alpha / (alpha + beta);
            let expected_variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));

            assert_close(stats.mean(), expected_mean, 0.05, "Beta mean");
            assert_close(stats.variance(), expected_variance, 0.10, "Beta variance");
        }
    }

    mod beta_tv {
        use super::*;

        #[test]
        fn tv_samples_positive() {
            let dist = BetaTV::new(
                |t| 1.0 + t.as_secs_float() * 0.5,
                |t| 2.0 + t.as_secs_float() * 0.5,
                TimeUnit::Seconds,
            );
            let mut rng = StdRng::from_os_rng();
            for i in 0..10 {
                let t = Duration::from_secs(i);
                let val = dist.sample(t, &mut rng).as_secs_float();
                assert!(
                    val >= 0.0 && val <= 1.0,
                    "Sample {val} not in [0,1] at t={t:?}"
                );
            }
        }

        #[test]
        #[ignore] // statistical test, expensive
        fn mean_and_variance_time_varying() {
            let dist = BetaTV::new(
                |t| 1.0 + t.as_secs_float() * 0.5,
                |t| 2.0 + t.as_secs_float() * 0.5,
                TimeUnit::Seconds,
            );
            let mut rng = StdRng::from_os_rng();
            const N: usize = 500_000;
            let time_points = [0, 5, 10, 20];

            for &t_sec in &time_points {
                let t = Duration::from_secs(t_sec);
                let alpha_t = 1.0 + t.as_secs_float() * 0.5;
                let beta_t = 2.0 + t.as_secs_float() * 0.5;

                let samples: Vec<Float> = (0..N)
                    .map(|_| dist.sample(t, &mut rng).as_secs_float())
                    .collect();

                let stats = BasicStatistics::compute(&samples);
                let expected_mean = alpha_t / (alpha_t + beta_t);
                let expected_variance =
                    (alpha_t * beta_t) / ((alpha_t + beta_t).powi(2) * (alpha_t + beta_t + 1.0));

                assert_close(
                    stats.mean(),
                    expected_mean,
                    0.05,
                    &format!("BetaTV mean at t={t_sec}s"),
                );
                assert_close(
                    stats.variance(),
                    expected_variance,
                    0.10,
                    &format!("BetaTV variance at t={t_sec}s"),
                );
            }
        }
    }
}
