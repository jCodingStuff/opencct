//! Weibull distribution

use std::time::Duration;
use rand::{
    rngs::StdRng, Rng, SeedableRng
};

use crate::{
    time::{DurationExtension, TimeUnit},
    Float,
};
use super::Distribution;

pub struct Weibull {
    /// Scale parameter
    lambda  : Float,
    /// Shape parameter
    k       : Float,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
}

/// Weibull distribution.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, Weibull};
/// use opencct::time::TimeUnit;
///
/// let mut dist = Weibull::new(1.0, 3.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {:?}", sample);
/// ```
impl Weibull {
    /// Create a new [Weibull] distribution with given shape and scale values.
    /// # Arguments
    /// * `lambda` - Scale
    /// * `k` - Shape
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [Weibull].
    /// # Panic
    /// This function panics if either `lambda` or `k` are <= 0
    pub fn new(lambda: Float, k: Float, unit: TimeUnit) -> Self {
        assert!(lambda > 0.0 && k > 0.0, "Invalid paramters: [lambda: {lambda}, k: {k}]");
        Self { lambda, k, factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    /// Create a new [Weibull] distribution with given random seed
    /// # Arguments
    /// * `lambda` - Scale
    /// * `k` - Shape
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// * A new [Weibull].
    /// # Panic
    /// This function panics if either `lambda` or `k` are <= 0
    pub fn new_seeded(lambda: Float, k: Float, unit: TimeUnit, seed: u64) -> Self {
        assert!(lambda > 0.0 && k > 0.0, "Invalid paramters: [lambda: {lambda}, k: {k}]");
        Self { lambda, k, factor: unit.factor(), rng: StdRng::seed_from_u64(seed) }
    }
}

impl Distribution for Weibull {
    fn sample(&mut self, _: Duration) -> Duration {
        let raw = self.lambda * (-self.rng.random::<Float>().ln()).powf(1.0 / self.k);
        Duration::from_secs_float(raw * self.factor)
    }
}

/// Weibull distribution with time-varying shape and scale.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, WeibullTV};
/// use opencct::time::{TimeUnit, DurationExtension};
///
/// let mut dist = WeibullTV::new(|t| 1.0 + t.as_secs_float() * 0.1, |t| 3.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct WeibullTV<Fl, Fk> {
    /// Scale parameter as a function of time
    lambda  : Fl,
    /// Shape parameter as a function of time
    k       : Fk,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
}

impl<Fl, Fk> WeibullTV<Fl, Fk>
where
    Fl: Fn(Duration) -> Float,
    Fk: Fn(Duration) -> Float,
{
    /// Create a new [WeibullTV] distribution with given shape and scale functions.
    /// # Arguments
    /// * `lambda` - Function to compute the scale at a given time. Must be > 0 for any t >= 0
    /// * `k` - Function to compute the shape at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [WeibullTV].
    /// # Be careful!
    /// `lambda` and `k` values are not checked in release mode! Make sure you fulfill the contract!
    pub fn new(lambda: Fl, k: Fk, unit: TimeUnit) -> Self {
        Self { lambda, k, factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    /// Create a new [WeibullTV] distribution with given random seed
    /// # Arguments
    /// * `lambda` - Function to compute the scale at a given time. Must be > 0 for any t >= 0
    /// * `k` - Function to compute the shape at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// * A new [WeibullTV].
    /// # Be careful!
    /// `lambda` and `k` values are not checked in release mode! Make sure you fulfill the contract!
    pub fn new_seeded(lambda: Fl, k: Fk, unit: TimeUnit, seed: u64) -> Self {
        Self { lambda, k, factor: unit.factor(), rng: StdRng::seed_from_u64(seed) }
    }
}

impl<Fl, Fk> Distribution for WeibullTV<Fl, Fk>
where
    Fl: Fn(Duration) -> Float,
    Fk: Fn(Duration) -> Float,
{
    /// See [Distribution::sample]
    /// # Panic
    /// In debug, this function will panic if at the requested time the shape or scale are <= 0.
    /// **This is NOT checked in release mode!**
    fn sample(&mut self, at: Duration) -> Duration {
        let (lambda, k) = ((self.lambda)(at), (self.k)(at));
        debug_assert!(lambda > 0.0 && k > 0.0, "Invalid lambda {lambda} or k {k} bound at {at:?}");
        let raw = lambda * (-self.rng.random::<Float>().ln()).powf(1.0 / k);
        Duration::from_secs_float(raw * self.factor)
    }
}
