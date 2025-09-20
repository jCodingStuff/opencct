//! Triangular distribution

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

/// Triangular distribution.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, Triangular};
/// use opencct::time::TimeUnit;
///
/// let mut dist = Triangular::new(1.0, 4.0, 1.5, TimeUnit::Hours);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct Triangular {
    /// Lower limit
    a       : Float,
    /// Upper limit
    b       : Float,
    /// Mode
    c       : Float,
    /// (c-a)/(b-a)
    fc      : Float,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
}

impl Triangular {
    /// Create a new [Triangular] distribution with given low and upper bounds, and mode
    /// # Arguments
    /// * `a` - Lower limit
    /// * `b` - Upper limit
    /// * `c` - Mode
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [Triangular].
    /// # Panic
    /// This function panics if `a < 0` or `c < a` or `b < c` or `b <= a`
    pub fn new(a: Float, b: Float, c: Float, unit: TimeUnit) -> Self {
        assert!(a >= 0.0 && c >= a && b >= c && b > a, "Invalid parameters [a: {a}, b: {b}, c: {c}]");
        Self { a, b, c, fc: (c-a)/(b-a), factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    /// Create a new [Triangular] distribution with given seed
    /// # Arguments
    /// * `a` - Lower limit
    /// * `b` - Upper limit
    /// * `c` - Mode
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// * A new [Triangular].
    /// # Panic
    /// This function panics if `a < 0` or `c < a` or `b < c` or `b <= a`
    pub fn new_seeded(a: Float, b: Float, c: Float, unit: TimeUnit, seed: u64) -> Self {
        assert!(a >= 0.0 && c >= a && b >= c && b > a, "Invalid parameters [a: {a}, b: {b}, c: {c}]");
        Self { a, b, c, fc: (c-a)/(b-a), factor: unit.factor(), rng: StdRng::seed_from_u64(seed) }
    }
}

impl Distribution for Triangular {
    fn sample(&mut self, _: Duration) -> Duration {
        let u = self.rng.random::<Float>();
        let raw = if u < self.fc {
            self.a + (u*(self.b-self.a)*(self.c-self.a)).sqrt()
        } else {
            self.b - ((1.0-u)*(self.b-self.a)*(self.b-self.c)).sqrt()
        };
        Duration::from_secs_float(raw * self.factor)
    }
}

/// Triangular distribution with time-varying parameters.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, TriangularTV};
/// use opencct::time::{TimeUnit, DurationExtension};
///
/// let mut dist = TriangularTV::new(
///     |t| 1.0 + t.as_hours_float() * 0.1,
///     |t| 3.0 + t.as_hours_float() * 0.1,
///     |t| 2.0 + t.as_hours_float() * 0.1,
///     TimeUnit::Hours,
/// );
/// let sample = dist.sample(Duration::from_hours(2));
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct TriangularTV<Fa, Fb, Fc> {
    /// Lower limit as a function of time
    a       : Fa,
    /// Upper limit as a function of time
    b       : Fb,
    /// Mode as a function of time
    c       : Fc,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
}

impl<Fa, Fb, Fc> TriangularTV<Fa, Fb, Fc>
where
    Fa: Fn(Duration) -> Float,
    Fb: Fn(Duration) -> Float,
    Fc: Fn(Duration) -> Float,
{
    /// Create a new [TriangularTV] distribution with given low and upper bounds, and mode functions
    /// # Arguments
    /// * `a` - Lower limit as a function of time. Must be >= 0 for any t >= 0
    /// * `b` - Upper limit as a function of time. Must be >= c and > a for any t >= 0
    /// * `c` - Mode as a function of time. Must be >= a for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [TriangularTV].
    /// # Be careful!
    /// `a`, `b`, and `c` conditions are not checked in release mode! Make sure you fulfill them!
    pub fn new(a: Fa, b: Fb, c: Fc, unit: TimeUnit) -> Self {
        Self { a, b, c, factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    /// Create a new [TriangularTV] distribution with given seed
    /// # Arguments
    /// * `a` - Lower limit as a function of time. Must be >= 0 for any t >= 0
    /// * `b` - Upper limit as a function of time. Must be >= c and > a for any t >= 0
    /// * `c` - Mode as a function of time. Must be >= a for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// * A new [TriangularTV].
    /// # Be careful!
    /// `a`, `b`, and `c` conditions are not checked in release mode! Make sure you fulfill them!
    pub fn new_seeded(a: Fa, b: Fb, c: Fc, unit: TimeUnit, seed: u64) -> Self {
        Self { a, b, c, factor: unit.factor(), rng: StdRng::seed_from_u64(seed) }
    }
}

impl<Fa, Fb, Fc> Distribution for TriangularTV<Fa, Fb, Fc>
where
    Fa: Fn(Duration) -> Float,
    Fb: Fn(Duration) -> Float,
    Fc: Fn(Duration) -> Float,
{
    /// See [Distribution::sample]
    /// # Panic
    /// In debug, this function will panic if at the requested time `a < 0` or `c < a` or `b < c` or `b < a`
    /// **This is NOT checked in release mode!**
    fn sample(&mut self, at: Duration) -> Duration {
        let (a, b, c) = ((self.a)(at), (self.b)(at), (self.c)(at));
        debug_assert!(a >= 0.0 && c >= a && b >= c && b > a, "At t:{at:?} found invalid parameters [a: {a}, b: {b}, c: {c}]");
        let u = self.rng.random::<Float>();
        let raw = if u < (c-a)/(b-a) {
            a + (u*(b-a)*(c-a)).sqrt()
        } else {
            b - ((1.0-u)*(b-a)*(b-c)).sqrt()
        };
        Duration::from_secs_float(raw * self.factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{BasicStatistics, assert_close};

    #[test]
    fn smoke_test_sampling() {
        let mut dist = Triangular::new(1.0, 5.0, 3.0, TimeUnit::Seconds);

        for _ in 0..10 {
            let v = dist.sample_at_t0().as_secs_float();
            assert!(v.is_finite(), "Sampled value must be finite, got {v}");
            assert!(v >= 1.0 && v <= 5.0, "Sample {v} out of bounds [1.0, 5.0]");
        }
    }

    #[test]
    #[should_panic]
    fn constructor_panics_a_negative() {
        Triangular::new(-1.0, 2.0, 1.0, TimeUnit::Seconds);
    }

    #[test]
    #[should_panic]
    fn constructor_panics_c_less_than_a() {
        Triangular::new(1.0, 3.0, 0.5, TimeUnit::Seconds);
    }

    #[test]
    #[should_panic]
    fn constructor_panics_b_less_than_c() {
        Triangular::new(1.0, 2.0, 3.0, TimeUnit::Seconds);
    }

    #[test]
    #[should_panic]
    fn constructor_panics_a_equals_b() {
        Triangular::new(2.0, 2.0, 2.0, TimeUnit::Seconds);
    }

    #[test]
    fn reproducibility_with_seed() {
        let mut d1 = Triangular::new_seeded(1.0, 3.0, 2.0, TimeUnit::Seconds, 42);
        let mut d2 = Triangular::new_seeded(1.0, 3.0, 2.0, TimeUnit::Seconds, 42);

        for _ in 0..100 {
            assert_eq!(d1.sample_at_t0(), d2.sample_at_t0());
        }
    }

    #[test]
    #[ignore]
    fn mean_and_variance() {
        const N_SAMPLES: usize = 100_000;
        let a = 1.0;
        let b = 5.0;
        let c = 3.0;

        let mut dist = Triangular::new_seeded(a, b, c, TimeUnit::Seconds, 123);
        let samples: Vec<Float> = (0..N_SAMPLES)
            .map(|_| dist.sample_at_t0().as_secs_float())
            .collect();

        let stats = BasicStatistics::compute(&samples);
        let (mean, var) = (stats.mean(), stats.variance());

        let theoretical_mean = (a + b + c) / 3.0;
        let theoretical_var = (a*a + b*b + c*c - a*b - a*c - b*c) / 18.0;

        assert_close(mean, theoretical_mean, 0.02, "Triangular mean"); // 2%
        assert_close(var, theoretical_var, 0.05, "Triangular variance"); // 5%
    }
}

#[cfg(test)]
mod tests_tv {
    use super::*;
    use crate::test_utils::{BasicStatistics, assert_close};

    #[test]
    fn smoke_test_sampling() {
        let mut dist = TriangularTV::new(|_| 1.0, |_| 5.0, |_| 3.0, TimeUnit::Seconds);

        for _ in 0..10 {
            let v = dist.sample_at_t0().as_secs_float();
            assert!(v.is_finite(), "Sampled value must be finite, got {v}");
            assert!(v >= 1.0 && v <= 5.0, "Sample {v} out of bounds [1.0, 5.0]");
        }
    }

    #[test]
    fn reproducibility_with_seed() {
        let mut d1 = TriangularTV::new_seeded(|_| 1.0, |_| 3.0, |_| 2.0, TimeUnit::Seconds, 42);
        let mut d2 = TriangularTV::new_seeded(|_| 1.0, |_| 3.0, |_| 2.0, TimeUnit::Seconds, 42);

        for _ in 0..100 {
            assert_eq!(d1.sample_at_t0(), d2.sample_at_t0());
        }
    }

    #[test]
    #[ignore]
    fn mean_and_variance() {
        const N_SAMPLES: usize = 100_000;
        let a = 1.0;
        let b = 5.0;
        let c = 3.0;

        let mut dist = TriangularTV::new_seeded(|_| a, |_| b, |_| c, TimeUnit::Seconds, 123);
        let samples: Vec<Float> = (0..N_SAMPLES)
            .map(|_| dist.sample_at_t0().as_secs_float())
            .collect();

        let stats = BasicStatistics::compute(&samples);
        let (mean, var) = (stats.mean(), stats.variance());

        let theoretical_mean = (a + b + c) / 3.0;
        let theoretical_var = (a*a + b*b + c*c - a*b - a*c - b*c) / 18.0;

        assert_close(mean, theoretical_mean, 0.02, "TriangularTV mean"); // 2%
        assert_close(var, theoretical_var, 0.05, "TriangularTV variance"); // 5%
    }
}
