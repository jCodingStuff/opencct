//! Beta distribution

use std::{time::Duration};
use rand::RngCore;

use crate::{
    time::TimeUnit,
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
/// use opencct::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = Beta::new(1.0, 3.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0(&mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Beta {
    /// Shape parameter
    alpha           : Float,
    /// Shape parameter
    beta            : Float,
    /// Time unit
    unit            : TimeUnit,
    /// Sampling method struct for alpha
    method_alpha    : MarsagliaTsang,
    /// Sampling method struct for beta
    method_beta     : MarsagliaTsang,
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
            alpha,
            beta,
            unit,
            method_alpha    : MarsagliaTsang::setup(alpha),
            method_beta     : MarsagliaTsang::setup(beta),
        }
    }
}

impl Distribution for Beta {
    fn sample(&self, _: Duration, rng: &mut dyn RngCore) -> Duration {
        let x = self.method_alpha.sample_from_setup(rng, 1.0);
        let y = self.method_beta.sample_from_setup(rng, 1.0);
        self.unit.to(x / (x + y))
    }

    fn mean(&self, _: Duration) -> Duration {
        self.unit.to(self.alpha / (self.alpha + self.beta))
    }

    fn variance(&self, _: Duration) -> Duration {
        self.unit.to2(
            self.alpha * self.beta / ((self.alpha + self.beta).powi(2) * (self.alpha + self.beta + 1.0))
        )
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
/// use opencct::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = BetaTV::new(
///     |t| 1.0 + TimeUnit::Seconds.from(t) * 0.1,
///     |t| 3.0 + TimeUnit::Seconds.from(t) * 0.1,
///     TimeUnit::Seconds,
/// );
/// let sample = dist.sample(Duration::from_secs(10), &mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct BetaTV<Fa, Fb> {
    /// Shape parameter
    alpha   : Fa,
    /// Shape parameter
    beta    : Fb,
    /// Time unit
    unit    : TimeUnit,
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
        Self { alpha, beta, unit }
    }

    /// Get the parameters (alpha, beta) of the distribution at a given point in time
    fn get_parameters_at(&self, at: Duration) -> (Float, Float) {
        let (alpha, beta) = ((self.alpha)(at), (self.beta)(at));
        debug_assert!(alpha > 0.0 && beta > 0.0, "Invalid alpha {alpha} or beta {beta} bound at {at:?}");
        (alpha, beta)
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
        let (alpha, beta) = self.get_parameters_at(at);
        let x = MarsagliaTsang::sample(rng, alpha, 1.0);
        let y = MarsagliaTsang::sample(rng, beta, 1.0);
        self.unit.to(x / (x + y))
    }

    /// See [Distribution::mean]
    /// # Panic
    /// In debug, this function will panic if at the requested time either of the shape parameters is <= 0.
    /// **This is NOT checked in release mode!**
    fn mean(&self, at: Duration) -> Duration {
        let (alpha, beta) = self.get_parameters_at(at);
        self.unit.to(alpha / (alpha + beta))
    }

    /// See [Distribution::variance]
    /// # Panic
    /// In debug, this function will panic if at the requested time either of the shape parameters is <= 0.
    /// **This is NOT checked in release mode!**
    fn variance(&self, at: Duration) -> Duration {
        let (alpha, beta) = self.get_parameters_at(at);
        self.unit.to2(alpha * beta / ((alpha + beta).powi(2) * (alpha + beta + 1.0)))
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
                let val = TimeUnit::Seconds.from(dist.sample_at_t0(&mut rng));
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

            let samples = dist.sample_n_at_t0(N, &mut rng);

            let stats = BasicStatistics::compute(&samples);

            assert_close(stats.mean(), dist.mean_at_t0(), 0.05, "Beta mean");
            assert_close(stats.variance(), dist.variance_at_t0(), 0.10, "Beta variance");
        }
    }

    mod beta_tv {
        use super::*;

        #[test]
        fn tv_samples_positive() {
            let dist = BetaTV::new(
                |t| 1.0 + TimeUnit::Seconds.from(t) * 0.5,
                |t| 2.0 + TimeUnit::Seconds.from(t) * 0.5,
                TimeUnit::Seconds,
            );
            let mut rng = StdRng::from_os_rng();
            for i in 0..10 {
                let t = Duration::from_secs(i);
                let val = TimeUnit::Seconds.from(dist.sample(t, &mut rng));
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
                |t| 1.0 + TimeUnit::Seconds.from(t) * 0.5,
                |t| 2.0 + TimeUnit::Seconds.from(t) * 0.5,
                TimeUnit::Seconds,
            );
            let mut rng = StdRng::from_os_rng();
            const N: usize = 500_000;

            for t_sec in [0, 5, 10, 20] {
                let t = Duration::from_secs(t_sec);

                let samples = dist.sample_n(N, t, &mut rng);

                let stats = BasicStatistics::compute(&samples);

                assert_close(
                    stats.mean(),
                    dist.mean(t),
                    0.05,
                    &format!("BetaTV mean at t={t_sec}s"),
                );
                assert_close(
                    stats.variance(),
                    dist.variance(t),
                    0.10,
                    &format!("BetaTV variance at t={t_sec}s"),
                );
            }
        }
    }
}
