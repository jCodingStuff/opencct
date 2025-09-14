//! Uniform distribution

use std::time::Duration;
use rand::{
    rngs::StdRng,
    Rng,
    SeedableRng,
};

use crate::Float;
use super::Distribution;

/// Uniform distribution.
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, Uniform};
///
/// let mut dist = Uniform::new(1.0, 3.0);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {}", sample);
/// ```
pub struct Uniform {
    /// Minimum value
    min: Float,
    /// Maximum value
    max: Float,
    /// Random number generator
    rng: StdRng,
}

impl Uniform {
    /// Create a new [Uniform] distribution with given minimum and maximum values.
    /// # Arguments
    /// * `min` - Minimum value.
    /// * `max` - Maximum value.
    /// # Returns
    /// * A new [Uniform].
    /// # Panic
    /// This function panics if `min > max` or if `min` or `max` is not positive
    pub fn new(min: Float, max: Float) -> Self {
        assert!(min <= max && min > 0.0 && max > 0.0, "Invalid range [{min}, {max}]");
        Self { min, max, rng: StdRng::from_os_rng() }
    }

    /// Create a new [Uniform] distribution with a specified random seed.
    /// # Arguments
    /// * `min` - Minimum value.
    /// * `max` - Maximum value.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// A new [Uniform].
    /// # Panic
    /// This function panics if `min > max` or if `min` or `max` is not positive
    pub fn new_seeded(min: Float, max: Float, seed: u64 ) -> Self {
        assert!(min <= max && min > 0.0 && max > 0.0, "Invalid range [{min}, {max}]");
        Self { min, max, rng: StdRng::seed_from_u64(seed) }
    }
}

impl Distribution for Uniform {
    fn sample(&mut self, _: Duration) -> Float {
        self.min + (self.max - self.min) * self.rng.random::<Float>()
    }
}

/// Uniform distribution with time-varying bounds.
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, UniformTV};
/// use opencct::DurationExtension;
///
/// let mut dist = UniformTV::new(|t| 1.0 + t.as_secs_float() * 0.1, |t| 3.0 + t.as_secs_float() * 0.1);
/// let sample = dist.sample(Duration::from_secs(10));
/// println!("Sampled value: {}", sample);
/// ```
pub struct UniformTV<FMin, FMax> {
    /// Minimum value as a function of time
    min: FMin,
    /// Maximum value as a function of time
    max: FMax,
    /// Random number generator
    rng: StdRng,
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
    /// # Returns
    /// A new [UniformTV].
    /// # Be careful!
    /// `min` and `max` bounds are not checked in release mode! Make sure you fulfill the bounds!
    pub fn new(min: FMin, max: FMax) -> Self {
        Self { min, max, rng: StdRng::from_os_rng() }
    }

    /// Create a new [UniformTV] distribution with a specified random seed.
    /// # Arguments
    /// * `min` - Function to compute the minimum bound at a given time. Must be > 0 and <= max for any t >= 0
    /// * `max` - Function to compute the maximum bound at a given time. Must be > 0 and >= min for any t >= 0
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// A new [UniformTV].
    /// # Be careful!
    /// `min` and `max` bounds are not checked in release mode! Make sure you fulfill the bounds!
    pub fn new_seeded(min: FMin, max: FMax, seed: u64 ) -> Self {
        Self { min, max, rng: StdRng::seed_from_u64(seed) }
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
    fn sample(&mut self, at: Duration) -> Float {
        let min = (self.min)(at);
        let max = (self.max)(at);
        debug_assert!(min <= max && min > 0.0 && max > 0.0, "Invalid bound at {at:?}: [{min}, {max}]");
        min + (max - min) * self.rng.random::<Float>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_within_bounds() {
        let low = 1.0;
        let high = 3.0;
        let mut dist = Uniform::new(low, high);

        for _ in 0..100 {
            let value = dist.sample_at_t0();
            assert!(
                value >= low && value <= high,
                "Value {} out of bounds [{}, {}]",
                value, low, high
            );
        }
    }

    #[test]
    fn seeded_reproducible() {
        let seed = 12345;
        let mut dist1 = Uniform::new_seeded(0.1, 1.0, seed);
        let mut dist2 = Uniform::new_seeded(0.1, 1.0, seed);

        for _ in 0..100 {
            let val1 = dist1.sample_at_t0();
            let val2 = dist2.sample_at_t0();
            assert_eq!(
                val1, val2,
                "Values {} and {} should be equal with the same seed",
                val1, val2
            );
        }
    }

    #[test]
    #[should_panic]
    fn invalid_bounds_panics() {
        let _ = Uniform::new(5.0, 2.0); // min > max should panic
    }

    #[test]
    fn min_equals_max_returns_constant() {
        let value = 7.5;
        let mut dist = Uniform::new(value, value);

        for _ in 0..10 {
            let sample = dist.sample_at_t0();
            assert_eq!(sample, value, "Sample should equal the fixed value");
        }
    }

    #[test]
    #[ignore]
    fn uniform_mean_approximation() {
        let min = 1.0;
        let max = 3.0;
        let mut dist = Uniform::new(min, max);

        let n_samples = 100_000;
        let mean: Float = (0..n_samples)
            .map(|_| dist.sample_at_t0())
            .sum::<Float>() / n_samples as Float;

        let expected_mean = (min + max) / 2.0;
        assert!((mean - expected_mean).abs() < 0.01,
            "Sample mean {} not close to expected {}", mean, expected_mean);
    }
}


#[cfg(test)]
mod tests_tv {
    use crate::DurationExtension;
    use super::*;

    #[test]
    fn samples_within_bounds() {
        let low = 1.0;
        let high = 3.0;
        let mut dist = UniformTV::new(|_| low, |_| high);

        for _ in 0..100 {
            let value = dist.sample_at_t0();
            assert!(value >= low && value <= high, "Value {} out of bounds [{}, {}]", value, low, high);
        }
    }

    #[test]
    fn time_dependent_bounds() {
        let offset = 5.0;
        let mut dist = UniformTV::new(
            |t| t.as_secs_float() + offset / 2.0,
            |t| t.as_secs_float() + offset,
        );

        for i in 0..5 {
            let t = Duration::from_secs(i);
            let value = dist.sample(t);
            assert!(
                value >= i as Float && value <= i as Float + offset,
                "At time {:?}, value {} out of bounds [{}, {}]",
                t, value, i as Float, i as Float + offset)
        }
    }

    #[test]
    fn seeded_reproducible() {
        let seed = 42;
        let mut dist1 = UniformTV::new_seeded(|_| 0.1, |_| 1.0, seed);
        let mut dist2 = UniformTV::new_seeded(|_| 0.1, |_| 1.0, seed);

        for _ in 0..100 {
            let val1 = dist1.sample_at_t0();
            let val2 = dist2.sample_at_t0();
            assert_eq!(val1, val2, "Values {} and {} should be equal with the same seed", val1, val2);
        }
    }

    #[test]
    #[should_panic]
    fn invalid_bounds() {
        let mut dist = UniformTV::new(|_| 5.0, |_| 2.0);
        dist.sample_at_t0();
    }

    #[test]
    fn min_equals_max_returns_constant() {
        let value = 4.2;
        let mut dist = UniformTV::new(|_| value, |_| value);

        for i in 0..10 {
            let t = Duration::from_secs(i);
            let sample = dist.sample(t);
            assert_eq!(sample, value, "At time {:?}, sample {} should equal the fixed value {}", t, sample, value);
        }
    }

    #[test]
    #[ignore]
    fn uniform_tv_mean_approximation() {
        let offset = 2.0;
        let mut dist = UniformTV::new(
            |t| 1.0 + t.as_secs_float() * 0.1,
            |t| 1.0 + t.as_secs_float() * 0.1 + offset,
        );

        let n_samples = 100_000;
        // Test at a few time points
        for t_sec in [0, 5, 10] {
            let t = Duration::from_secs(t_sec);
            let mean: Float = (0..n_samples)
                .map(|_| dist.sample(t))
                .sum::<Float>() / n_samples as Float;

            let min = 1.0 + t_sec as Float * 0.1;
            let max = min + offset;
            let expected_mean = (min + max) / 2.0;

            assert!((mean - expected_mean).abs() < 0.01,
                "At t={}, sample mean {} not close to expected {}", t_sec, mean, expected_mean);
        }
    }

}
