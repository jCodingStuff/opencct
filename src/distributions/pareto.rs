//! Pareto distribution

use std::time::Duration;
use rand::{
    Rng,
    rngs::StdRng,
    SeedableRng,
};

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
/// use opencct::distributions::{Distribution, Pareto};
/// use opencct::time::TimeUnit;
///
/// let mut dist = Pareto::new(1.0, 3.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct Pareto {
    /// Scale parameter
    xm      : Float,
    /// Shape parameter
    alpha   : Float,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
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
        Self { xm, alpha, factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    /// Create a new [Pareto] distribution with given random seed.
    /// # Arguments
    /// * `xm` - Scale
    /// * `alpha` - Shape
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// * A new [Pareto].
    /// # Panic
    /// This function panics if either `xm` or `alpha` are <= 0
    pub fn new_seeded(xm: Float, alpha: Float, unit: TimeUnit, seed: u64) -> Self {
        assert!(xm > 0.0 && alpha > 0.0, "Invalid xm {xm} or alpha {alpha}");
        Self { xm, alpha, factor: unit.factor(), rng: StdRng::seed_from_u64(seed) }
    }
}

impl Distribution for Pareto {
    fn sample(&mut self, _: Duration) -> Duration {
        let raw = self.xm / self.rng.random::<Float>().powf(1.0 / self.alpha);
        Duration::from_secs_float(raw * self.factor)
    }
}

/// Pareto distribution with time-varying shape and scale.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, ParetoTV};
/// use opencct::time::{TimeUnit, DurationExtension};
///
/// let mut dist = ParetoTV::new(|t| 1.0 + t.as_secs_float() * 0.1, |t| 3.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct ParetoTV<Fx, Fa> {
    /// Scale parameter
    xm      : Fx,
    /// Shape parameter
    alpha   : Fa,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
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
        Self { xm, alpha, factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    /// Create a new [ParetoTV] distribution with given random seed.
    /// # Arguments
    /// * `xm` - Function to compute the scale at a given time. Must be > 0 for any t >= 0
    /// * `alpha` - Function to compute the shape at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// * A new [ParetoTV].
    /// # Be careful!
    /// `xm` and `alpha` values are not checked in release mode! Make sure you fulfill the contract!
    pub fn new_seeded(xm: Fx, alpha: Fa, unit: TimeUnit, seed: u64) -> Self {
        Self { xm, alpha, factor: unit.factor(), rng: StdRng::seed_from_u64(seed) }
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
    fn sample(&mut self, at: Duration) -> Duration {
        let (xm, alpha) = ((self.xm)(at), (self.alpha)(at));
        debug_assert!(xm > 0.0 && alpha > 0.0, "Invalid xm {xm} or alpha {alpha} bound at {at:?}");
        let raw = xm / self.rng.random::<Float>().powf(1.0 / alpha);
        Duration::from_secs_float(raw * self.factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{TimeUnit, DurationExtension};
    use crate::test_utils::{BasicStatistics, assert_close};

    #[test]
    fn samples_above_xm() {
        let mut dist = Pareto::new(1.0, 3.0, TimeUnit::Seconds);
        for _ in 0..100 {
            let val = dist.sample_at_t0().as_secs_float();
            assert!(val >= 1.0, "Sample {val} should be >= xm");
        }
    }

    #[test]
    fn seeded_reproducible() {
        let mut dist1 = Pareto::new_seeded(1.0, 2.5, TimeUnit::Seconds, 42);
        let mut dist2 = Pareto::new_seeded(1.0, 2.5, TimeUnit::Seconds, 42);
        for _ in 0..100 {
            let v1 = dist1.sample_at_t0().as_secs_float();
            let v2 = dist2.sample_at_t0().as_secs_float();
            assert_eq!(v1, v2, "Seeded distributions diverged: {v1} vs {v2}");
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
        let mut dist = Pareto::new(xm, alpha, TimeUnit::Seconds);
        let n = 200_000;
        let samples: Vec<Float> = (0..n)
            .map(|_| dist.sample_at_t0().as_secs_float())
            .collect();

        let stats = BasicStatistics::compute(&samples);
        let expected_mean = alpha * xm / (alpha - 1.0);
        let expected_var = (xm * xm * alpha) / ((alpha - 1.0).powi(2) * (alpha - 2.0));

        assert_close(stats.mean(), expected_mean, 0.05, "Pareto mean");
        assert_close(stats.variance(), expected_var, 0.10, "Pareto variance");
    }

    #[test]
    fn tv_samples_positive() {
        let mut dist = ParetoTV::new(|t| 1.0 + t.as_secs_f64() * 0.1, |_| 2.0, TimeUnit::Seconds);
        for i in 0..10 {
            let t = Duration::from_secs(i);
            let val = dist.sample(t).as_secs_float();
            let expected_xm = 1.0 + i as Float * 0.1;
            assert!(
                val >= expected_xm,
                "Sample {val} below xm {expected_xm} at t={t:?}"
            );
        }
    }

    #[test]
    fn tv_seeded_reproducible() {
        let mut dist1 = ParetoTV::new_seeded(|_| 1.0, |_| 2.5, TimeUnit::Seconds, 99);
        let mut dist2 = ParetoTV::new_seeded(|_| 1.0, |_| 2.5, TimeUnit::Seconds, 99);
        for _ in 0..100 {
            let v1 = dist1.sample_at_t0().as_secs_float();
            let v2 = dist2.sample_at_t0().as_secs_float();
            assert_eq!(v1, v2, "TV seeded distributions diverged: {v1} vs {v2}");
        }
    }
}
