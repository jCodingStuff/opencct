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
    pub fn new(a: Float, b: Float, c: Float, unit: TimeUnit) -> Self {
        assert!(a >= 0.0 && c >= a && b >= c, "Invalid parameters [a: {a}, b: {b}, c: {c}] ");
        Self { a, b, c, fc: (c-a)/(b-a), factor: unit.factor(), rng: StdRng::from_os_rng() }
    }

    pub fn new_seeded(a: Float, b: Float, c: Float, unit: TimeUnit, seed: u64) -> Self {
        assert!(a >= 0.0 && c >= a && b >= c, "Invalid parameters [a: {a}, b: {b}, c: {c}] ");
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
