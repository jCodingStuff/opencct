//! Triangular distribution

use std::time::Duration;
use rand::{Rng, RngCore};

use crate::{
    time::TimeUnit,
    Float,
};
use super::Distribution;

/// Triangular distribution.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, Triangular};
/// use opencct::time::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = Triangular::new(1.0, 4.0, 1.5, TimeUnit::Hours);
/// let sample = dist.sample_at_t0(&mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Triangular {
    /// Lower limit
    a       : Float,
    /// Upper limit
    b       : Float,
    /// Mode
    c       : Float,
    /// (c-a)/(b-a)
    fc      : Float,
    /// Time unit
    unit    : TimeUnit,
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
        Self { a, b, c, fc: (c-a)/(b-a), unit }
    }
}

impl Distribution for Triangular {
    fn sample(&self, _: Duration, rng: &mut dyn RngCore) -> Duration {
        let u = rng.random::<Float>();
        let raw = if u < self.fc {
            self.a + (u*(self.b-self.a)*(self.c-self.a)).sqrt()
        } else {
            self.b - ((1.0-u)*(self.b-self.a)*(self.b-self.c)).sqrt()
        };
        self.unit.to_duration(raw)
    }

    fn mean(&self, _: Duration) -> Duration {
        let raw = (self.a + self.b + self.c) / 3.0;
        self.unit.to_duration(raw)
    }

    fn variance(&self, _: Duration) -> Duration {
        let raw = (
            self.a.powi(2) + self.b.powi(2) + self.c.powi(2)
            - self.a * self.b - self.a * self.c - self.b * self.c
        ) / 18.0;
        self.unit.to_duration(raw)
    }
}

/// Triangular distribution with time-varying parameters.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, TriangularTV};
/// use opencct::time::TimeUnit;
///
/// let mut rng = StdRng::from_os_rng();
/// let dist = TriangularTV::new(
///     |t| 1.0 + TimeUnit::Seconds.from_duration(t) * 0.1,
///     |t| 3.0 + TimeUnit::Seconds.from_duration(t) * 0.1,
///     |t| 2.0 + TimeUnit::Seconds.from_duration(t) * 0.1,
///     TimeUnit::Hours,
/// );
/// let sample = dist.sample(Duration::from_hours(2), &mut rng);
/// println!("Sampled value: {:?}", sample);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct TriangularTV<Fa, Fb, Fc> {
    /// Lower limit as a function of time
    a       : Fa,
    /// Upper limit as a function of time
    b       : Fb,
    /// Mode as a function of time
    c       : Fc,
    /// Time unit
    unit    : TimeUnit,
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
        Self { a, b, c, unit }
    }

    /// Get the parameters (a, b, c) of the distribution at a given point in time
    fn get_parameters_at(&self, at: Duration) -> (Float, Float, Float) {
        let (a, b, c) = ((self.a)(at), (self.b)(at), (self.c)(at));
        debug_assert!(a >= 0.0 && c >= a && b >= c && b > a, "At t:{at:?} found invalid parameters [a: {a}, b: {b}, c: {c}]");
        (a, b, c)
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
    fn sample(&self, at: Duration, rng: &mut dyn RngCore) -> Duration {
        let (a, b, c) = self.get_parameters_at(at);
        let u = rng.random::<Float>();
        let raw = if u < (c-a)/(b-a) {
            a + (u*(b-a)*(c-a)).sqrt()
        } else {
            b - ((1.0-u)*(b-a)*(b-c)).sqrt()
        };
        self.unit.to_duration(raw)
    }

    /// See [Distribution::mean]
    /// # Panic
    /// In debug, this function will panic if at the requested time `a < 0` or `c < a` or `b < c` or `b < a`
    /// **This is NOT checked in release mode!**
    fn mean(&self, at: Duration) -> Duration {
        let (a, b, c) = self.get_parameters_at(at);
        let raw = (a + b + c) / 3.0;
        self.unit.to_duration(raw)
    }

    /// See [Distribution::variance]
    /// # Panic
    /// In debug, this function will panic if at the requested time `a < 0` or `c < a` or `b < c` or `b < a`
    /// **This is NOT checked in release mode!**
    fn variance(&self, at: Duration) -> Duration {
        let (a, b, c) = self.get_parameters_at(at);
        let raw = (a.powi(2) + b.powi(2) + c.powi(2) - a * b - a * c - b * c) / 18.0;
        self.unit.to_duration(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use crate::test_utils::{BasicStatistics, assert_close};

    mod triangular {
        use super::*;

        #[test]
        fn smoke_test_sampling() {
            let dist = Triangular::new(1.0, 5.0, 3.0, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            for _ in 0..10 {
                let v = TimeUnit::Seconds.from_duration(dist.sample_at_t0(&mut rng));
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
        #[ignore]
        fn mean_and_variance() {
            const N_SAMPLES: usize = 500_000;
            let a = 1.0;
            let b = 5.0;
            let c = 3.0;

            let dist = Triangular::new(a, b, c, TimeUnit::Seconds);
            let mut rng = StdRng::seed_from_u64(123);
            let samples = dist.sample_n_at_t0(N_SAMPLES, &mut rng);

            let stats = BasicStatistics::compute(&samples);

            assert_close(stats.mean(), dist.mean_at_t0(), 0.02, "Triangular mean");
            assert_close(stats.variance(), dist.variance_at_t0(), 0.05, "Triangular variance");
        }
    }

    mod triangular_tv {
        use super::*;

        #[test]
        fn smoke_test_sampling() {
            let dist = TriangularTV::new(|_| 1.0, |_| 5.0, |_| 3.0, TimeUnit::Seconds);
            let mut rng = StdRng::from_os_rng();

            for _ in 0..10 {
                let v = TimeUnit::Seconds.from_duration(dist.sample_at_t0(&mut rng));
                assert!(v.is_finite(), "Sampled value must be finite, got {v}");
                assert!(v >= 1.0 && v <= 5.0, "Sample {v} out of bounds [1.0, 5.0]");
            }
        }

        #[test]
        #[ignore] // statistical test, expensive
        fn mean_and_variance_large_sample_tv() {
            const N_SAMPLES: usize = 500_000;

            let dist = TriangularTV::new(
                |t| 0.5 * TimeUnit::Seconds.from_duration(t) + 0.5,
                |t| 0.7 * TimeUnit::Seconds.from_duration(t) + 4.0,
                |t| 0.65 * TimeUnit::Seconds.from_duration(t) + 0.89,
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
                    &format!("TriangularTV mean at t={t_sec}"),
                );
                assert_close(
                    stats.variance(),
                    dist.variance(t),
                    0.02,
                    &format!("TriangularTV variance at t={t_sec}"),
                );
            }
        }
    }
}
