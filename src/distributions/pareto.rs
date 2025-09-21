//! Pareto distribution

use std::time::Duration;
use rand::Rng;

use crate::{
    time::{DurationExtension, TimeUnit},
    Float,
};
use super::Distribution;

/// Pareto distribution.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, Pareto};
/// use opencct::time::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = Pareto::new(1.0, 3.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0(&mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Pareto {
    /// Scale parameter
    xm      : Float,
    /// Shape parameter
    alpha   : Float,
    /// Time unit factor
    factor  : Float,
}

impl Pareto {
    /// Create a new [Pareto] distribution with given shape and scale values.
    /// # Arguments
    /// * `xm` - Scale
    /// * `alpha` - Shape
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [Pareto].
    /// # Panic
    /// This function panics if either `xm` or `alpha` are <= 0
    pub fn new(xm: Float, alpha: Float, unit: TimeUnit) -> Self {
        assert!(xm > 0.0 && alpha > 0.0, "Invalid xm {xm} or alpha {alpha}");
        Self { xm, alpha, factor: unit.factor() }
    }
}

impl Distribution for Pareto {
    fn sample<R: Rng + ?Sized>(&self, _: Duration, rng: &mut R) -> Duration {
        let raw = self.xm / rng.random::<Float>().powf(1.0 / self.alpha);
        Duration::from_secs_float(raw * self.factor)
    }
}

/// Pareto distribution with time-varying shape and scale.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, ParetoTV};
/// use opencct::time::{TimeUnit, DurationExtension};
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = ParetoTV::new(|t| 1.0 + t.as_secs_float() * 0.1, |t| 3.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
/// let sample = dist.sample(Duration::from_secs(10), &mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct ParetoTV<Fx, Fa> {
    /// Scale parameter
    xm      : Fx,
    /// Shape parameter
    alpha   : Fa,
    /// Time unit factor
    factor  : Float,
}

impl<Fx, Fa> ParetoTV<Fx, Fa>
where
    Fx: Fn(Duration) -> Float,
    Fa: Fn(Duration) -> Float,
{
    /// Create a new [ParetoTV] distribution with given shape and scale functions.
    /// # Arguments
    /// * `xm` - Function to compute the scale at a given time. Must be > 0 for any t >= 0
    /// * `alpha` - Function to compute the shape at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [ParetoTV].
    /// # Be careful!
    /// `xm` and `alpha` values are not checked in release mode! Make sure you fulfill the contract!
    pub fn new(xm: Fx, alpha: Fa, unit: TimeUnit) -> Self {
        Self { xm, alpha, factor: unit.factor() }
    }
}

impl<Fx, Fa> Distribution for ParetoTV<Fx, Fa>
where
    Fx: Fn(Duration) -> Float,
    Fa: Fn(Duration) -> Float,
{
    /// See [Distribution::sample]
    /// # Panic
    /// In debug, this function will panic if at the requested time the shape or scale are <= 0.
    /// **This is NOT checked in release mode!**
    fn sample<R: Rng + ?Sized>(&self, at: Duration, rng: &mut R) -> Duration {
        let (xm, alpha) = ((self.xm)(at), (self.alpha)(at));
        debug_assert!(xm > 0.0 && alpha > 0.0, "Invalid xm {xm} or alpha {alpha} bound at {at:?}");
        let raw = xm / rng.random::<Float>().powf(1.0 / alpha);
        Duration::from_secs_float(raw * self.factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use crate::test_utils::{BasicStatistics, assert_close};

    mod pareto {
        use super::*;

        #[test]
        fn samples_above_xm() {
            let dist = Pareto::new(1.0, 3.0, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();
            for _ in 0..100 {
                let val = dist.sample_at_t0(&mut rng).as_secs_float();
                assert!(val >= 1.0, "Sample {val} should be >= xm");
            }
        }

        #[test]
        #[should_panic]
        fn invalid_xm_panics() {
            let _ = Pareto::new(0.0, 1.0, TimeUnit::Seconds);
        }

        #[test]
        #[should_panic]
        fn invalid_alpha_panics() {
            let _ = Pareto::new(1.0, 0.0, TimeUnit::Seconds);
        }

        #[test]
        #[ignore]
        fn mean_and_variance() {
            let xm = 1.0;
            let alpha = 3.0;
            let dist = Pareto::new(xm, alpha, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();
            let n = 500_000;
            let samples: Vec<Float> = (0..n)
                .map(|_| dist.sample_at_t0(&mut rng).as_secs_float())
                .collect();

            let stats = BasicStatistics::compute(&samples);
            let expected_mean = alpha * xm / (alpha - 1.0);
            let expected_var =
                (xm * xm * alpha) / ((alpha - 1.0).powi(2) * (alpha - 2.0));

            assert_close(stats.mean(), expected_mean, 0.05, "Pareto mean");
            assert_close(stats.variance(), expected_var, 0.10, "Pareto variance");
        }
    }

    mod pareto_tv {
        use super::*;

        #[test]
        fn samples_positive() {
            let dist = ParetoTV::new(
                |t| 1.0 + t.as_secs_f64() * 0.1,
                |_| 2.0,
                TimeUnit::Seconds,
            );
            let mut rng = StdRng::from_os_rng();
            for i in 0..10 {
                let t = Duration::from_secs(i);
                let val = dist.sample(t, &mut rng).as_secs_float();
                let expected_xm = 1.0 + i as Float * 0.1;
                assert!(
                    val >= expected_xm,
                    "Sample {val} below xm {expected_xm} at t={t:?}"
                );
            }
        }

        #[test]
        #[ignore]
        fn mean_and_variance_time_varying() {
            let dist = ParetoTV::new(
                |t| 1.0 + t.as_secs_f64() * 0.2,
                |_| 3.0,
                TimeUnit::Seconds,
            );
            let mut rng = StdRng::from_os_rng();
            let n = 500_000;

            // test at multiple time points
            for &secs in &[0_u64, 5, 10, 20] {
                let t = Duration::from_secs(secs);
                let xm = 1.0 + t.as_secs_f64() * 0.2;
                let alpha = 3.0;

                let samples: Vec<Float> =
                    (0..n).map(|_| dist.sample(t, &mut rng).as_secs_float()).collect();

                let stats = BasicStatistics::compute(&samples);
                let expected_mean = alpha * xm / (alpha - 1.0);
                let expected_var =
                    (xm * xm * alpha) / ((alpha - 1.0).powi(2) * (alpha - 2.0));

                assert_close(
                    stats.mean(),
                    expected_mean,
                    0.05,
                    &format!("ParetoTV mean at t={secs}"),
                );
                assert_close(
                    stats.variance(),
                    expected_var,
                    0.10,
                    &format!("ParetoTV variance at t={secs}"),
                );
            }
        }
    }
}
