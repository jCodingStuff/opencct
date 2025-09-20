//! Beta distribution

use std::time::Duration;
use rand::{
    rngs::StdRng,
    SeedableRng,
};

use crate::{
    time::{DurationExtension, TimeUnit},
    Float,
};
use super::{
    Distribution,
    algorithms::marsaglia_tsang::MarsagliaTsang,
};

pub struct Beta {
    /// Shape parameter
    alpha: Float,
    /// Shape parameter
    beta: Float,
    /// Time unit factor
    factor: Float,
    /// Random number generator
    rng: StdRng,
    /// Sampling method struct for alpha
    method_alpha: MarsagliaTsang,
    /// Sampling method struct for beta
    method_beta: MarsagliaTsang,
}

impl Beta {
    pub fn new(alpha: Float, beta: Float, unit: TimeUnit) -> Self {
        assert!(alpha > 0.0 && beta > 0.0, "Invalid alpha {alpha} or beta {beta}");
        Self {
            alpha,
            beta,
            factor          : unit.factor(),
            rng             : StdRng::from_os_rng(),
            method_alpha    : MarsagliaTsang::setup(alpha),
            method_beta     : MarsagliaTsang::setup(beta),
        }
    }

    pub fn new_seeded(alpha: Float, beta: Float, unit: TimeUnit, seed: u64) -> Self {
        assert!(alpha > 0.0 && beta > 0.0, "Invalid alpha {alpha} or beta {beta}");
        Self {
            alpha,
            beta,
            factor          : unit.factor(),
            rng             : StdRng::seed_from_u64(seed),
            method_alpha    : MarsagliaTsang::setup(alpha),
            method_beta     : MarsagliaTsang::setup(beta),
        }
    }
}

impl Distribution for Beta {
    fn sample(&mut self, _: Duration) -> Duration {
        let x = self.method_alpha.sample_from_setup(&mut self.rng, 1.0);
        let y = self.method_beta.sample_from_setup(&mut self.rng, 1.0);
        Duration::from_secs_float(x / (x + y) * self.factor)
    }
}

pub struct BetaTV<Fa, Fb> {
    /// Shape parameter
    alpha: Fa,
    /// Shape parameter
    beta: Fb,
    /// Time unit factor
    factor: Float,
    /// Random number generator
    rng: StdRng,
}
