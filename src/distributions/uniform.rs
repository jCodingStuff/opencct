//! Uniform distribution

use std::time::Duration;
use rand::{
    rngs::StdRng,
    Rng,
    SeedableRng,
};

use super::Distribution;

/// Uniform distribution with time-varying bounds.
/// # Type Parameters
/// * `F` - A function type that takes a `Duration` and returns a `f64`.
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, Uniform};
///
/// let mut dist = Uniform::new(|t| 1.0 + t.as_secs_f64() * 0.1, |t| 3.0 + t.as_secs_f64() * 0.1);
/// let sample = dist.sample(Duration::from_secs(10));
/// println!("Sampled value: {}", sample);
/// ```
pub struct Uniform<FMin, FMax>
where
    FMin: Fn(Duration) -> f64,
    FMax: Fn(Duration) -> f64,
{
    min: FMin,
    max: FMax,
    rng: StdRng,
}

impl<FMin, FMax> Uniform<FMin, FMax>
where
    FMin: Fn(Duration) -> f64,
    FMax: Fn(Duration) -> f64,
{
    /// Create a new `Uniform` distribution with given min and max functions.
    /// # Arguments
    /// * `min` - Function to compute the minimum bound at a given time.
    /// * `max` - Function to compute the maximum bound at a given time.
    /// # Returns
    /// * `Self` - A new instance of `Uniform`.
    pub fn new(min: FMin, max: FMax) -> Self
    {
        Self {
            min,
            max,
            rng: StdRng::from_os_rng()
        }
    }

    /// Create a new `Uniform` distribution with a specified random seed.
    /// # Arguments
    /// * `min` - Function to compute the minimum bound at a given time.
    /// * `max` - Function to compute the maximum bound at a given time.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// * `Self` - A new instance of `Uniform`.
    pub fn new_seeded(min: FMin, max: FMax, seed: u64 ) -> Self
    {
        Self {
            min,
            max,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl<FMin, FMax> Distribution for Uniform<FMin, FMax>
where
    FMin: Fn(Duration) -> f64,
    FMax: Fn(Duration) -> f64,
{
    fn sample(&mut self, at: Duration) -> f64 {
        let min = (self.min)(at);
        let max = (self.max)(at);
        assert!(min <= max);
        min + (max - min) * self.rng.random::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_samples_within_bounds() {
        let low = 1.0;
        let high = 3.0;
        let mut dist = Uniform::new(|_| low, |_| high);

        for _ in 0..100 {
            let value = dist.sample(Duration::from_secs(0));
            assert!(value >= low && value <= high, "Value {} out of bounds [{}, {}]", value, low, high);
        }
    }

    #[test]
    fn uniform_time_dependent_bounds() {
        let offset = 5.0;
        let mut dist = Uniform::new(
            |t| t.as_secs_f64(),
            |t| t.as_secs_f64() + offset,
        );

        for i in 0..5 {
            let t = Duration::from_secs(i);
            let value = dist.sample(t);
            assert!(
                value >= i as f64 && value <= i as f64 + offset,
                "At time {:?}, value {} out of bounds [{}, {}]",
                t, value, i as f64, i as f64 + offset)
        }
    }

    #[test]
    fn uniform_seeded_reproducible() {
        let seed = 42;
        let mut dist1 = Uniform::new_seeded(|_| 0.0, |_| 1.0, seed);
        let mut dist2 = Uniform::new_seeded(|_| 0.0, |_| 1.0, seed);

        for _ in 0..100 {
            let val1 = dist1.sample(Duration::from_secs(0));
            let val2 = dist2.sample(Duration::from_secs(0));
            assert_eq!(val1, val2, "Values {} and {} should be equal with the same seed", val1, val2);
        }
    }

    #[test]
    #[should_panic]
    fn uniform_invalid_bounds() {
        let mut dist = Uniform::new(|_| 5.0, |_| 2.0);
        dist.sample(Duration::from_secs(0));
    }

}
