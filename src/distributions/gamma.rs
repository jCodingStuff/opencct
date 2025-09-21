//! Gamma-Erlang distribution
//!
//! Erlang is just a Gamma with an integer shape parameter

use std::time::Duration;
use rand::Rng;

use crate::{
    time::{DurationExtension, TimeUnit},
    Float,
};
use super::{
    Distribution,
    algorithms::marsaglia_tsang::MarsagliaTsang,
};

/// Gamma-Erlang distribution.
///
/// Implemented via the Marsaglia-Tsang method. See
/// Marsaglia, G., & Tsang, W. W. (2000).
/// [A simple method for generating gamma variables](https://doi.acm.org/10.1145/358407.358414).
/// *ACM Transactions on Mathematical Software (TOMS)*, 26(3), 363-372.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, GammaErlang};
/// use opencct::time::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = GammaErlang::new(1.0, 3.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0(&mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct GammaErlang {
    /// Scale parameter
    theta   : Float,
    /// Time unit factor
    factor  : Float,
    /// Sampling method struct
    method  : MarsagliaTsang,
}

impl GammaErlang {
    /// Create a new [GammaErlang] distribution with given shape and scale values.
    /// # Arguments
    /// * `alpha` - Shape
    /// * `theta` - Scale
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [GammaErlang].
    /// # Panic
    /// This function panics if either `alpha` or `theta` are <= 0
    pub fn new(alpha: Float, theta: Float, unit: TimeUnit) -> Self {
        assert!(alpha > 0.0 && theta > 0.0, "Invalid alpha {alpha} or theta {theta}");
        Self {theta, factor: unit.factor(), method: MarsagliaTsang::setup(alpha) }
    }
}

impl Distribution for GammaErlang {
    fn sample<R: Rng + ?Sized>(&self, _: Duration, rng: &mut R) -> Duration {
        let raw = self.method.sample_from_setup(rng, self.theta);
        Duration::from_secs_float(raw * self.factor)
    }
}

/// Gamma-Erlang distribution with time-varying shape and scale.
///
/// Implemented via the Marsaglia-Tsang method. See
/// Marsaglia, G., & Tsang, W. W. (2000).
/// [A simple method for generating gamma variables](https://doi.acm.org/10.1145/358407.358414).
/// *ACM Transactions on Mathematical Software (TOMS)*, 26(3), 363-372.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, GammaErlangTV};
/// use opencct::time::{TimeUnit, DurationExtension};
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = GammaErlangTV::new(|t| 1.0 + t.as_secs_float() * 0.1, |t| 3.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
/// let sample = dist.sample(Duration::from_secs(10), &mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct GammaErlangTV<Fa, Fb> {
    /// Shape parameter as a function of time
    alpha   : Fa,
    /// Scale parameter as a function of time
    theta   : Fb,
    /// Time unit factor
    factor  : Float,
}

impl<Fa, Fb> GammaErlangTV<Fa, Fb>
where
    Fa: Fn(Duration) -> Float,
    Fb: Fn(Duration) -> Float,
{
    /// Create a new [GammaErlang] distribution with given shape and scale functions.
    /// # Arguments
    /// * `alpha` - Function to compute the shape at a given time. Must be > 0 for any t >= 0
    /// * `theta` - Function to compute the scale at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [GammaErlang].
    /// # Be careful!
    /// `alpha` and `theta` values are not checked in release mode! Make sure you fulfill the contract!
    pub fn new(alpha: Fa, theta: Fb, unit: TimeUnit) -> Self {
        Self { alpha, theta, factor: unit.factor() }
    }
}

impl<Fa, Fb> Distribution for GammaErlangTV<Fa, Fb>
where
    Fa: Fn(Duration) -> Float,
    Fb: Fn(Duration) -> Float,
{
    /// See [Distribution::sample]
    /// # Panic
    /// In debug, this function will panic if at the requested time the shape or scale are <= 0.
    /// **This is NOT checked in release mode!**
    fn sample<R: Rng + ?Sized>(&self, at: Duration, rng: &mut R) -> Duration {
        let alpha = (self.alpha)(at);
        let theta = (self.theta)(at);
        debug_assert!(alpha > 0.0 && theta > 0.0, "Invalid alpha {alpha} or theta {theta} bound at {at:?}");
        let raw = MarsagliaTsang::sample(rng, alpha, theta);
        Duration::from_secs_float(raw * self.factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use crate::test_utils::{BasicStatistics, assert_close};

    mod gamma_erlang {
        use super::*;

        #[test]
        #[should_panic]
        fn new_panics_on_zero_alpha() {
            let _ = GammaErlang::new(0.0, 1.0, TimeUnit::Seconds);
        }

        #[test]
        #[should_panic]
        fn new_panics_on_zero_theta() {
            let _ = GammaErlang::new(1.0, 0.0, TimeUnit::Seconds);
        }

        #[test]
        fn smoke_sample() {
            let dist = GammaErlang::new(2.0, 3.0, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();
            let _ = dist.sample_at_t0(&mut rng);
        }

        #[test]
        fn values_are_finite() {
            let dist = GammaErlang::new(3.0, 2.0, TimeUnit::Seconds);
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

            let alpha = 2.0;
            let theta = 3.0;
            let dist = GammaErlang::new(alpha, theta, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            let samples: Vec<Float> = (0..N_SAMPLES)
                .map(|_| dist.sample_at_t0(&mut rng).as_secs_float())
                .collect();

            let stats = BasicStatistics::compute(&samples);
            let expected_mean = alpha * theta;
            let expected_var = alpha * theta.powi(2);

            assert_close(stats.mean(), expected_mean, 0.01, "GammaErlang mean"); // 1% tolerance
            assert_close(stats.variance(), expected_var, 0.02, "GammaErlang variance"); // 2% tolerance
        }
    }

    mod gamma_erlang_tv {
        use super::*;

        #[test]
        fn smoke_sample_tv() {
            let dist = GammaErlangTV::new(|_| 2.0, |_| 3.0, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();
            let _ = dist.sample_at_t0(&mut rng);
            let _ = dist.sample(Duration::from_secs(5), &mut rng);
        }

        #[test]
        #[ignore] // statistical test, expensive
        fn mean_and_variance_large_sample_tv() {
            const N_SAMPLES: usize = 100_000;

            let alpha = 2.0;
            let theta = 3.0;
            let dist = GammaErlangTV::new(|_| alpha, |_| theta, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            for t_sec in [0, 5, 10] {
                let t = Duration::from_secs(t_sec);
                let samples: Vec<Float> = (0..N_SAMPLES)
                    .map(|_| dist.sample(t, &mut rng).as_secs_float())
                    .collect();

                let stats = BasicStatistics::compute(&samples);
                let expected_mean = alpha * theta;
                let expected_var = alpha * theta.powi(2);

                assert_close(
                    stats.mean(),
                    expected_mean,
                    0.01,
                    &format!("GammaErlangTV mean at t={t_sec}"),
                );
                assert_close(
                    stats.variance(),
                    expected_var,
                    0.02,
                    &format!("GammaErlangTV variance at t={t_sec}"),
                );
            }
        }
    }
}
