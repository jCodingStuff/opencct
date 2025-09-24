//! Pareto distribution

use std::time::Duration;
use rand::{Rng, RngCore};

use crate::{
    time::TimeUnit,
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
    /// Time unit
    unit    : TimeUnit,
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
        Self { xm, alpha, unit }
    }
}

impl Distribution for Pareto {
    fn sample(&self, _: Duration, rng: &mut dyn RngCore) -> Duration {
        let raw = self.xm / rng.random::<Float>().powf(1.0 / self.alpha);
        self.unit.to_duration(raw)
    }

    fn mean(&self, _: Duration) -> Duration {
        let raw = if self.alpha <= 1.0 { Float::INFINITY } else { self.alpha * self.xm / (self.alpha - 1.0) };
        self.unit.to_duration(raw)
    }

    fn variance(&self, _: Duration) -> Duration {
        let raw = if self.alpha <= 2.0 {
            Float::INFINITY
        } else {
            self.xm.powi(2) * self.alpha / ((self.alpha - 1.0).powi(2) * (self.alpha - 2.0))
        };
        self.unit.to_duration(raw)
    }
}

/// Pareto distribution with time-varying shape and scale.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, ParetoTV};
/// use opencct::time::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = ParetoTV::new(
///     |t| 1.0 + TimeUnit::Seconds.from_duration(t) * 0.1,
///     |t| 3.0 + TimeUnit::Seconds.from_duration(t) * 0.1,
///     TimeUnit::Seconds,
/// );
/// let sample = dist.sample(Duration::from_secs(10), &mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct ParetoTV<Fx, Fa> {
    /// Scale parameter
    xm      : Fx,
    /// Shape parameter
    alpha   : Fa,
    /// Time unit
    unit    : TimeUnit,
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
        Self { xm, alpha, unit }
    }

    /// Get the parameters (xm, alpha) of the distribution at a given point in time
    fn get_parameters_at(&self, at: Duration) -> (Float, Float) {
        let (xm, alpha) = ((self.xm)(at), (self.alpha)(at));
        debug_assert!(xm > 0.0 && alpha > 0.0, "Invalid xm {xm} or alpha {alpha} bound at {at:?}");
        (xm, alpha)
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
    fn sample(&self, at: Duration, rng: &mut dyn RngCore) -> Duration {
        let (xm, alpha) = self.get_parameters_at(at);
        let raw = xm / rng.random::<Float>().powf(1.0 / alpha);
        self.unit.to_duration(raw)
    }

    /// See [Distribution::mean]
    /// # Panic
    /// In debug, this function will panic if at the requested time the shape or scale are <= 0.
    /// **This is NOT checked in release mode!**
    fn mean(&self, at: Duration) -> Duration {
        let (xm, alpha) = self.get_parameters_at(at);
        let raw = if alpha <= 1.0 { Float::INFINITY } else { alpha * xm / (alpha - 1.0) };
        self.unit.to_duration(raw)
    }

    /// See [Distribution::variance]
    /// # Panic
    /// In debug, this function will panic if at the requested time the shape or scale are <= 0.
    /// **This is NOT checked in release mode!**
    fn variance(&self, at: Duration) -> Duration {
        let (xm, alpha) = self.get_parameters_at(at);
        let raw = if alpha <= 2.0 {
            Float::INFINITY
        } else {
            xm.powi(2) * alpha / ((alpha - 1.0).powi(2) * (alpha - 2.0))
        };
        self.unit.to_duration(raw)
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
                let val = TimeUnit::Seconds.from_duration(dist.sample_at_t0(&mut rng));
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
            const N: usize = 500_000;
            let samples = dist.sample_n_at_t0(N, &mut rng);

            let stats = BasicStatistics::compute(&samples);

            assert_close(stats.mean(), dist.mean_at_t0(), 0.05, "Pareto mean");
            assert_close(stats.variance(), dist.variance_at_t0(), 0.10, "Pareto variance");
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
                let val = TimeUnit::Seconds.from_duration(dist.sample(t, &mut rng));
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
            const N: usize = 500_000;

            // test at multiple time points
            for secs in [0, 5, 10, 20] {
                let t = Duration::from_secs(secs);
                let samples = dist.sample_n(N, t, &mut rng);

                let stats = BasicStatistics::compute(&samples);

                assert_close(
                    stats.mean(),
                    dist.mean(t),
                    0.05,
                    &format!("ParetoTV mean at t={secs}"),
                );
                assert_close(
                    stats.variance(),
                    dist.variance(t),
                    0.10,
                    &format!("ParetoTV variance at t={secs}"),
                );
            }
        }
    }
}
