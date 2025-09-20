//! Uniform distribution

use std::time::Duration;
use rand::{
    rngs::StdRng,
    Rng,
    SeedableRng,
};

use crate::{
    time::{DurationExtension, TimeUnit},
    Float,
};
use super::Distribution;

/// Uniform distribution.
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, Uniform};
/// use opencct::time::TimeUnit;
///
/// let mut dist = Uniform::new(1.0, 3.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct Uniform {
    /// Minimum value
    min     : Float,
    /// Maximum value
    max     : Float,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
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
        Self { min, max, factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    /// Create a new [Uniform] distribution with a specified random seed.
    /// # Arguments
    /// * `min` - Minimum value.
    /// * `max` - Maximum value.
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// A new [Uniform].
    /// # Panic
    /// This function panics if `min > max` or if `min` or `max` is not positive
    pub fn new_seeded(min: Float, max: Float, unit: TimeUnit, seed: u64) -> Self {
        assert!(min <= max && min >= 0.0, "Invalid range [{min}, {max}]");
        Self { min, max, factor: unit.factor(), rng: StdRng::seed_from_u64(seed) }
    }
}

impl Distribution for Uniform {
    fn sample(&mut self, _: Duration) -> Duration {
        let raw = self.min + (self.max - self.min) * self.rng.random::<Float>();
        Duration::from_secs_float(raw * self.factor)
    }
}

/// Uniform distribution with time-varying bounds.
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, UniformTV};
/// use opencct::time::{TimeUnit, DurationExtension};
///
/// let mut dist = UniformTV::new(|t| 1.0 + t.as_secs_float() * 0.1, |t| 3.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
/// let sample = dist.sample(Duration::from_secs(10));
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct UniformTV<FMin, FMax> {
    /// Minimum value as a function of time
    min     : FMin,
    /// Maximum value as a function of time
    max     : FMax,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
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
        Self { min, max, factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    /// Create a new [UniformTV] distribution with a specified random seed.
    /// # Arguments
    /// * `min` - Function to compute the minimum bound at a given time. Must be > 0 and <= max for any t >= 0
    /// * `max` - Function to compute the maximum bound at a given time. Must be > 0 and >= min for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// A new [UniformTV].
    /// # Be careful!
    /// `min` and `max` bounds are not checked in release mode! Make sure you fulfill the bounds!
    pub fn new_seeded(min: FMin, max: FMax, unit: TimeUnit, seed: u64) -> Self {
        Self { min, max, factor: unit.factor(), rng: StdRng::seed_from_u64(seed) }
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
    fn sample(&mut self, at: Duration) -> Duration {
        let min = (self.min)(at);
        let max = (self.max)(at);
        debug_assert!(min <= max && min >= 0.0, "Invalid bound at {at:?}: [{min}, {max}]");
        let raw = min + (max - min) * self.rng.random::<Float>();
        Duration::from_secs_float(raw * self.factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{BasicStatistics, assert_close};

    #[test]
    fn samples_within_bounds() {
        let low = 1.0;
        let high = 3.0;
        let mut dist = Uniform::new(low, high, TimeUnit::Seconds);

        for _ in 0..100 {
            let sample = dist.sample_at_t0().as_secs_float();
            assert!(sample >= low && sample <= high, "Sample {sample} out of bounds [{low}, {high}]");
        }
    }

    #[test]
    fn seeded_reproducible() {
        let seed = 12345;
        let mut dist1 = Uniform::new_seeded(0.0, 1.0, TimeUnit::Seconds, seed);
        let mut dist2 = Uniform::new_seeded(0.0, 1.0, TimeUnit::Seconds, seed);

        for _ in 0..100 {
            assert_eq!(dist1.sample_at_t0(), dist2.sample_at_t0(), "Values with same seed should match");
        }
    }

    #[test]
    #[should_panic]
    fn invalid_bounds_panics() {
        let _ = Uniform::new(5.0, 2.0, TimeUnit::Seconds);
    }

    #[test]
    fn min_equals_max_returns_constant() {
        let value = 7.5;
        let mut dist = Uniform::new(value, value, TimeUnit::Seconds);

        for _ in 0..10 {
            let sample = dist.sample_at_t0().as_secs_float();
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

        for &unit in &units {
            let mut dist = Uniform::new(base_value, base_value + delta, unit);
            let sample = dist.sample_at_t0().as_secs_float();
            let factor = unit.factor();
            assert!(
                sample >= base_value * factor && sample <= (base_value + delta) * factor,
                "Uniform sample {:?} out of expected range [{}, {}] for unit {:?}",
                sample, base_value * factor, (base_value + delta) * factor, unit
            );
        }
    }

    #[test]
    #[ignore]
    fn mean_and_variance() {
        const N_SAMPLES: usize = 100_000;
        let low = 1.0;
        let high = 5.0;
        let mut dist = Uniform::new_seeded(low, high, TimeUnit::Seconds, 42);

        let samples: Vec<Float> = (0..N_SAMPLES)
            .map(|_| dist.sample_at_t0().as_secs_float())
            .collect();

        let stats = BasicStatistics::compute(&samples);
        let (mean, var) = (stats.mean(), stats.variance());

        let expected_mean = (low + high) / 2.0;
        let expected_var = ((high - low).powi(2)) / 12.0;

        assert_close(mean, expected_mean, 0.01, "Uniform mean"); // 1% tolerance
        assert_close(var, expected_var, 0.02, "Uniform variance"); // 2% tolerance
    }
}

#[cfg(test)]
mod tests_tv {
    use super::*;
    use crate::test_utils::{BasicStatistics, assert_close};

    #[test]
    fn samples_within_bounds() {
        let low = 1.0;
        let high = 3.0;
        let mut dist = UniformTV::new(|_| low, |_| high, TimeUnit::Seconds);

        for _ in 0..100 {
            let sample = dist.sample_at_t0().as_secs_float();
            assert!(sample >= low && sample <= high, "Sample {sample} out of bounds");
        }
    }

    #[test]
    fn time_dependent_bounds() {
        let offset = 5.0;
        let mut dist = UniformTV::new(
            |t| t.as_secs_float() + offset / 2.0,
            |t| t.as_secs_float() + offset,
            TimeUnit::Seconds
        );

        for i in 0..5 {
            let t = Duration::from_secs(i);
            let sample = dist.sample(t).as_secs_float();
            assert!(
                sample >= i as Float && sample <= i as Float + offset,
                "At time {:?}, sample {} out of bounds [{}, {}]",
                t, sample, i as Float, i as Float + offset
            );
        }
    }

    #[test]
    fn seeded_reproducible() {
        let seed = 42;
        let mut dist1 = UniformTV::new_seeded(|_| 0.0, |_| 1.0, TimeUnit::Seconds, seed);
        let mut dist2 = UniformTV::new_seeded(|_| 0.0, |_| 1.0, TimeUnit::Seconds, seed);

        for _ in 0..100 {
            assert_eq!(dist1.sample_at_t0(), dist2.sample_at_t0(), "Values with same seed should match");
        }
    }

    #[test]
    #[should_panic]
    fn invalid_bounds_panics() {
        let mut dist = UniformTV::new(|_| 5.0, |_| 2.0, TimeUnit::Seconds);
        dist.sample_at_t0();
    }

    #[test]
    fn min_equals_max_returns_constant() {
        let value = 4.2;
        let mut dist = UniformTV::new(|_| value, |_| value, TimeUnit::Seconds);

        for i in 0..10 {
            let t = Duration::from_secs(i);
            let sample = dist.sample(t).as_secs_float();
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

        for &unit in &units {
            let mut dist_tv = UniformTV::new(|_| base_value, |_| base_value + delta, unit);
            let sample_tv = dist_tv.sample(Duration::from_secs(0)).as_secs_float();
            let factor = unit.factor();
            assert!(
                sample_tv >= base_value * factor && sample_tv <= (base_value + delta) * factor,
                "UniformTV sample {:?} out of expected range [{}, {}] for unit {:?}",
                sample_tv, base_value * factor, (base_value + delta) * factor, unit
            );
        }
    }

    #[test]
    #[ignore]
    fn mean_and_variance() {
        const N_SAMPLES: usize = 100_000;
        let low = 1.0;
        let high = 5.0;
        let mut dist = UniformTV::new_seeded(|_| low, |_| high, TimeUnit::Seconds, 42);

        let t = Duration::from_secs(5);
        let samples: Vec<Float> = (0..N_SAMPLES)
            .map(|_| dist.sample(t).as_secs_float())
            .collect();

        let stats = BasicStatistics::compute(&samples);
        let (mean, var) = (stats.mean(), stats.variance());

        let expected_mean = (low + high) / 2.0;
        let expected_var = ((high - low).powi(2)) / 12.0;

        assert_close(mean, expected_mean, 0.01, "UniformTV mean"); // 1% tolerance
        assert_close(var, expected_var, 0.02, "UniformTV variance"); // 2% tolerance
    }
}
