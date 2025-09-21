//! Uniform distribution

use std::time::Duration;
use rand::{Rng, RngCore};

use crate::{
    time::{DurationExtension, TimeUnit},
    Float,
};
use super::Distribution;

/// Uniform distribution.
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, Uniform};
/// use opencct::time::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = Uniform::new(1.0, 3.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0(&mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Uniform {
    /// Minimum value
    min     : Float,
    /// Maximum value
    max     : Float,
    /// Time unit factor
    factor  : Float,
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
        Self { min, max, factor: unit.factor() }
    }
}

impl Distribution for Uniform {
    fn sample(&self, _: Duration, rng: &mut dyn RngCore) -> Duration {
        let raw = self.min + (self.max - self.min) * rng.random::<Float>();
        Duration::from_secs_float(raw * self.factor)
    }
}

/// Uniform distribution with time-varying bounds.
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, UniformTV};
/// use opencct::time::{TimeUnit, DurationExtension};
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = UniformTV::new(|t| 1.0 + t.as_secs_float() * 0.1, |t| 3.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
/// let sample = dist.sample(Duration::from_secs(10), &mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct UniformTV<FMin, FMax> {
    /// Minimum value as a function of time
    min     : FMin,
    /// Maximum value as a function of time
    max     : FMax,
    /// Time unit factor
    factor  : Float,
}

impl<FMin, FMax> UniformTV<FMin, FMax>
where
    FMin: Fn(Duration) -> Float,
    FMax: Fn(Duration) -> Float,
{
    /// Create a new [UniformTV] distribution with given min and max functions.
    /// # Arguments
    /// * `min` - Function to compute the minimum bound at a given time. Must be > 0 and <= max for any t >= 0
    /// * `max` - Function to compute the maximum bound at a given time. Must be > 0 and >= min for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// A new [UniformTV].
    /// # Be careful!
    /// `min` and `max` bounds are not checked in release mode! Make sure you fulfill the bounds!
    pub fn new(min: FMin, max: FMax, unit: TimeUnit) -> Self {
        Self { min, max, factor: unit.factor() }
    }
}

impl<FMin, FMax> Distribution for UniformTV<FMin, FMax>
where
    FMin: Fn(Duration) -> Float,
    FMax: Fn(Duration) -> Float,
{
    /// See [Distribution::sample]
    /// # Panic
    /// In debug, this function will panic if at the requested time the lower bound is higher than the upper one
    /// or if any of the bounds <= 0. **This is NOT checked in release mode!**
    fn sample(&self, at: Duration, rng: &mut dyn RngCore) -> Duration {
        let min = (self.min)(at);
        let max = (self.max)(at);
        debug_assert!(min <= max && min >= 0.0, "Invalid bound at {at:?}: [{min}, {max}]");
        let raw = min + (max - min) * rng.random::<Float>();
        Duration::from_secs_float(raw * self.factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use crate::test_utils::{BasicStatistics, assert_close};

    mod uniform {
        use super::*;

        #[test]
        fn samples_within_bounds() {
            let low = 1.0;
            let high = 3.0;
            let dist = Uniform::new(low, high, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            for _ in 0..100 {
                let sample = dist.sample_at_t0(&mut rng).as_secs_float();
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
                let sample = dist.sample_at_t0(&mut rng).as_secs_float();
                assert_close(sample, value, 0.0, "Uniform constant sample");
            }
        }

        #[test]
        fn time_units() {
            let base_value = 2.0;
            let delta = 1.0;
            let units = [
                TimeUnit::Days,
                TimeUnit::Hours,
                TimeUnit::Minutes,
                TimeUnit::Seconds,
                TimeUnit::Millis,
                TimeUnit::Nanos,
            ];
            let mut rng = StdRng::from_os_rng();

            for &unit in &units {
                let dist = Uniform::new(base_value, base_value + delta, unit);
                let sample = dist.sample_at_t0(&mut rng).as_secs_float();
                let factor = unit.factor();
                assert!(
                    sample >= base_value * factor && sample <= (base_value + delta) * factor,
                    "Uniform sample {sample:?} out of expected range [{}, {}] for unit {:?}",
                    base_value * factor,
                    (base_value + delta) * factor,
                    unit
                );
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

            let samples: Vec<Float> = (0..N_SAMPLES)
                .map(|_| dist.sample_at_t0(&mut rng).as_secs_float())
                .collect();

            let stats = BasicStatistics::compute(&samples);
            let expected_mean = (low + high) / 2.0;
            let expected_var = ((high - low).powi(2)) / 12.0;

            assert_close(stats.mean(), expected_mean, 0.01, "Uniform mean");
            assert_close(stats.variance(), expected_var, 0.02, "Uniform variance");
        }
    }

    mod uniform_tv {
        use super::*;

        #[test]
        fn samples_within_bounds() {
            let low = 1.0;
            let high = 3.0;
            let dist = UniformTV::new(|_| low, |_| high, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            for _ in 0..100 {
                let sample = dist.sample_at_t0(&mut rng).as_secs_float();
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
                |t| t.as_secs_float() + offset / 2.0,
                |t| t.as_secs_float() + offset,
                TimeUnit::Seconds,
            );
            let mut rng = StdRng::from_os_rng();

            for i in 0..5 {
                let t = Duration::from_secs(i);
                let sample = dist.sample(t, &mut rng).as_secs_float();
                assert!(
                    sample >= i as Float && sample <= i as Float + offset,
                    "At time {:?}, sample {} out of bounds [{}, {}]",
                    t,
                    sample,
                    i as Float,
                    i as Float + offset
                );
            }
        }

        #[test]
        #[should_panic]
        fn invalid_bounds_panics() {
            let dist = UniformTV::new(|_| 5.0, |_| 2.0, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();
            dist.sample_at_t0(&mut rng);
        }

        #[test]
        fn min_equals_max_returns_constant() {
            let value = 4.2;
            let dist = UniformTV::new(|_| value, |_| value, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            for i in 0..10 {
                let t = Duration::from_secs(i);
                let sample = dist.sample(t, &mut rng).as_secs_float();
                assert_close(sample, value, 0.0, "UniformTV constant sample");
            }
        }

        #[test]
        fn time_units() {
            let base_value = 2.0;
            let delta = 1.0;
            let units = [
                TimeUnit::Days,
                TimeUnit::Hours,
                TimeUnit::Minutes,
                TimeUnit::Seconds,
                TimeUnit::Millis,
                TimeUnit::Nanos,
            ];
            let mut rng = StdRng::from_os_rng();

            for &unit in &units {
                let dist = UniformTV::new(|_| base_value, |_| base_value + delta, unit);
                let sample = dist.sample(Duration::from_secs(0), &mut rng).as_secs_float();
                let factor = unit.factor();
                assert!(
                    sample >= base_value * factor && sample <= (base_value + delta) * factor,
                    "UniformTV sample {sample:?} out of expected range [{}, {}] for unit {:?}",
                    base_value * factor,
                    (base_value + delta) * factor,
                    unit
                );
            }
        }

        #[test]
        #[ignore]
        fn mean_and_variance() {
            const N_SAMPLES: usize = 500_000;
            let low = 1.0;
            let high = 5.0;
            let dist = UniformTV::new(|_| low, |_| high, TimeUnit::Seconds);
            let mut rng = StdRng::seed_from_u64(42);

            let t = Duration::from_secs(5);
            let samples: Vec<Float> = (0..N_SAMPLES)
                .map(|_| dist.sample(t, &mut rng).as_secs_float())
                .collect();

            let stats = BasicStatistics::compute(&samples);
            let expected_mean = (low + high) / 2.0;
            let expected_var = ((high - low).powi(2)) / 12.0;

            assert_close(stats.mean(), expected_mean, 0.01, "UniformTV mean");
            assert_close(stats.variance(), expected_var, 0.02, "UniformTV variance");
        }
    }
}
