//! Uniform distribution

use std::time::Duration;
use rand::{
    rngs::StdRng,
    Rng,
    SeedableRng,
};

use crate::Float;
use super::{Distribution};

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
    /// This function panics if `min > max`
    pub fn new(min: Float, max: Float) -> Self {
        assert!(min <= max, "Invalid range [{min}, {max}]");
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
    /// This function panics if `min > max`
    pub fn new_seeded(min: Float, max: Float, seed: u64 ) -> Self {
        assert!(min <= max, "Invalid range [{min}, {max}]");
        Self { min, max, rng: StdRng::seed_from_u64(seed) }
    }
}

impl Distribution for Uniform {
    fn sample(&mut self, _: Duration) -> Float {
        self.min + (self.max - self.min) * self.rng.random::<Float>()
    }
}

/// Uniform distribution with time-varying bounds.
/// # Type Parameters
/// * `F` - A function type that takes a [Duration] and returns a [Float].
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
pub struct UniformTV<FMin, FMax>
where
    FMin: Fn(Duration) -> Float,
    FMax: Fn(Duration) -> Float,
{
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
    /// * `min` - Function to compute the minimum bound at a given time.
    /// * `max` - Function to compute the maximum bound at a given time.
    /// # Returns
    /// A new `UniformTV`.
    pub fn new(min: FMin, max: FMax) -> Self {
        Self { min, max, rng: StdRng::from_os_rng() }
    }

    /// Create a new [UniformTV] distribution with a specified random seed.
    /// # Arguments
    /// * `min` - Function to compute the minimum bound at a given time.
    /// * `max` - Function to compute the maximum bound at a given time.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// A new [UniformTV].
    pub fn new_seeded(min: FMin, max: FMax, seed: u64 ) -> Self {
        Self { min, max, rng: StdRng::seed_from_u64(seed) }
    }
}

impl<FMin, FMax> Distribution for UniformTV<FMin, FMax>
where
    FMin: Fn(Duration) -> Float,
    FMax: Fn(Duration) -> Float,
{
    fn sample(&mut self, at: Duration) -> Float {
        let min = (self.min)(at);
        let max = (self.max)(at);
        assert!(min <= max);
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
        let mut dist1 = Uniform::new_seeded(0.0, 1.0, seed);
        let mut dist2 = Uniform::new_seeded(0.0, 1.0, seed);

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
            |t| t.as_secs_float(),
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
        let mut dist1 = UniformTV::new_seeded(|_| 0.0, |_| 1.0, seed);
        let mut dist2 = UniformTV::new_seeded(|_| 0.0, |_| 1.0, seed);

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

}
