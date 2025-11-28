//! Uniform distribution

use rand::{Rng, RngCore};
use std::time::Duration;

use super::{Distribution, TimeVaryingParameterFunction};
use crate::{Float, time::TimeUnit};

/// Uniform distribution.
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, Uniform};
/// use opencct::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = Uniform::new(1.0, 3.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0(&mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Uniform {
    /// Minimum value
    min: Float,
    /// Maximum value
    max: Float,
    /// Time unit
    unit: TimeUnit,
}

impl Uniform {
    /// Create a new [Uniform] distribution with given minimum and maximum values.
    /// # Arguments
    /// * `min` - Minimum value.
    /// * `max` - Maximum value.
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [Uniform].
    /// # Panic
    /// This function panics if `min > max` or if `min` or `max` is not positive
    pub fn new(min: Float, max: Float, unit: TimeUnit) -> Self {
        assert!(min <= max && min >= 0.0, "Invalid range [{min}, {max}]");
        Self { min, max, unit }
    }
}

impl Distribution for Uniform {
    fn sample(&self, _: Duration, rng: &mut dyn RngCore) -> Duration {
        let raw = self.min + (self.max - self.min) * rng.random::<Float>();
        self.unit.to(raw)
    }

    fn mean(&self, _: Duration) -> Duration {
        let raw = 0.5 * (self.min + self.max);
        self.unit.to(raw)
    }

    fn variance(&self, _: Duration) -> Duration {
        let raw = (self.max - self.min).powi(2) / 12.0;
        self.unit.to2(raw)
    }
}

/// Uniform distribution with time-varying bounds.
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, UniformTV};
/// use opencct::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = UniformTV::new(
///     Box::new(|t| 1.0 + TimeUnit::Seconds.from(t) * 0.1),
///     Box::new(|t| 3.0 + TimeUnit::Seconds.from(t) * 0.1),
///     TimeUnit::Seconds,
/// );
/// let sample = dist.sample(Duration::from_secs(10), &mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct UniformTV {
    /// Minimum value as a function of time
    min: TimeVaryingParameterFunction,
    /// Maximum value as a function of time
    max: TimeVaryingParameterFunction,
    /// Time unit
    unit: TimeUnit,
}

impl UniformTV {
    /// Create a new [UniformTV] distribution with given min and max functions.
    /// # Arguments
    /// * `min` - Function to compute the minimum bound at a given time. Must be > 0 and <= max for any t >= 0
    /// * `max` - Function to compute the maximum bound at a given time. Must be > 0 and >= min for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// A new [UniformTV].
    pub fn new(
        min: TimeVaryingParameterFunction,
        max: TimeVaryingParameterFunction,
        unit: TimeUnit,
    ) -> Self {
        Self { min, max, unit }
    }

    /// Get the bounds (min, max) of the distribution at a given point in time
    fn get_bounds_at(&self, at: Duration) -> (Float, Float) {
        let (min, max) = ((self.min)(at), (self.max)(at));
        assert!(
            min <= max && min >= 0.0,
            "Invalid bound at {at:?}: [{min}, {max}]"
        );
        (min, max)
    }
}

impl Distribution for UniformTV {
    /// See [Distribution::sample]
    /// # Panic
    /// This function will panic if at the requested time the lower bound is higher than the upper one
    /// or if any of the bounds <= 0.
    fn sample(&self, at: Duration, rng: &mut dyn RngCore) -> Duration {
        let (min, max) = self.get_bounds_at(at);
        let raw = min + (max - min) * rng.random::<Float>();
        self.unit.to(raw)
    }

    /// See [Distribution::mean]
    /// # Panic
    /// This function will panic if at the requested time the lower bound is higher than the upper one
    /// or if any of the bounds <= 0.
    fn mean(&self, at: Duration) -> Duration {
        let (min, max) = self.get_bounds_at(at);
        let raw = 0.5 * (min + max);
        self.unit.to(raw)
    }

    /// See [Distribution::variance]
    /// # Panic
    /// This function will panic if at the requested time the lower bound is higher than the upper one
    /// or if any of the bounds <= 0.
    fn variance(&self, at: Duration) -> Duration {
        let (min, max) = self.get_bounds_at(at);
        let raw = (max - min).powi(2) / 12.0;
        self.unit.to2(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{BasicStatistics, assert_close};
    use rand::{SeedableRng, rngs::StdRng};

    mod uniform {
        use super::*;

        #[test]
        fn samples_within_bounds() {
            let low = 1.0;
            let high = 3.0;
            let dist = Uniform::new(low, high, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            for _ in 0..100 {
                let sample = TimeUnit::Seconds.from(dist.sample_at_t0(&mut rng));
                assert!(
                    sample >= low && sample <= high,
                    "Sample {sample} out of bounds [{low}, {high}]"
                );
            }
        }

        #[test]
        #[should_panic]
        fn invalid_bounds_panics() {
            Uniform::new(5.0, 2.0, TimeUnit::Seconds);
        }

        #[test]
        fn min_equals_max_returns_constant() {
            let value = 7.5;
            let dist = Uniform::new(value, value, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            for _ in 0..10 {
                let sample = TimeUnit::Seconds.from(dist.sample_at_t0(&mut rng));
                assert_close(sample, value, 0.0, "Uniform constant sample");
            }
        }

        #[test]
        #[ignore]
        fn mean_and_variance() {
            const N_SAMPLES: usize = 100_000;
            let low = 1.0;
            let high = 5.0;
            let mut rng = StdRng::seed_from_u64(42);
            let dist = Uniform::new(low, high, TimeUnit::Seconds);

            let samples = dist.sample_n_at_t0(N_SAMPLES, &mut rng);

            let stats = BasicStatistics::compute(&samples);

            assert_close(stats.mean(), dist.mean_at_t0(), 0.01, "Uniform mean");
            assert_close(
                stats.variance(),
                dist.variance_at_t0(),
                0.02,
                "Uniform variance",
            );
        }
    }

    mod uniform_tv {
        use super::*;

        #[test]
        fn samples_within_bounds() {
            let low = 1.0;
            let high = 3.0;
            let dist = UniformTV::new(
                Box::new(move |_| low),
                Box::new(move |_| high),
                TimeUnit::Seconds,
            );
            let mut rng = StdRng::from_os_rng();

            for _ in 0..100 {
                let sample = TimeUnit::Seconds.from(dist.sample_at_t0(&mut rng));
                assert!(
                    sample >= low && sample <= high,
                    "Sample {sample} out of bounds"
                );
            }
        }

        #[test]
        fn time_dependent_bounds() {
            let offset = 5.0;
            let dist = UniformTV::new(
                Box::new(|t| TimeUnit::Seconds.from(t)),
                Box::new(move |t| TimeUnit::Seconds.from(t) + offset),
                TimeUnit::Seconds,
            );
            let mut rng = StdRng::from_os_rng();

            for i in 0..5 {
                let t = Duration::from_secs(i);
                let sample = dist.sample(t, &mut rng);
                let low = TimeUnit::Seconds.to(i as Float);
                let high = TimeUnit::Seconds.to(i as Float + offset);
                assert!(
                    sample >= low && sample <= high,
                    "At time {:?}, sample {:?} out of bounds [{:?}, {:?}]",
                    t,
                    sample,
                    low,
                    high,
                );
            }
        }

        #[test]
        #[should_panic]
        fn invalid_bounds_panics() {
            let dist = UniformTV::new(Box::new(|_| 5.0), Box::new(|_| 2.0), TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();
            dist.sample_at_t0(&mut rng);
        }

        #[test]
        fn min_equals_max_returns_constant() {
            let value = 4.2;
            let dist = UniformTV::new(
                Box::new(move |_| value),
                Box::new(move |_| value),
                TimeUnit::Seconds,
            );
            let mut rng = StdRng::from_os_rng();

            for i in 0..10 {
                let t = Duration::from_secs(i);
                let sample = dist.sample(t, &mut rng);
                assert_close(sample, value, 0.0, "UniformTV constant sample");
            }
        }

        #[test]
        #[ignore] // statistical test, expensive
        fn mean_and_variance_large_sample_tv() {
            const N_SAMPLES: usize = 500_000;

            let dist = UniformTV::new(
                Box::new(|t| 0.5 * TimeUnit::Seconds.from(t) + 0.1),
                Box::new(|t| 0.8 * TimeUnit::Seconds.from(t) + 1.0),
                TimeUnit::Seconds,
            );
            let mut rng = StdRng::from_os_rng();

            for t_sec in [0, 5, 10] {
                let t = Duration::from_secs(t_sec);
                let samples = dist.sample_n(N_SAMPLES, t, &mut rng);

                let stats = BasicStatistics::compute(&samples);

                assert_close(
                    stats.mean(),
                    dist.mean(t),
                    0.01,
                    &format!("UniformTV mean at t={t_sec}"),
                );
                assert_close(
                    stats.variance(),
                    dist.variance(t),
                    0.02,
                    &format!("UniformTV variance at t={t_sec}"),
                );
            }
        }
    }
}
