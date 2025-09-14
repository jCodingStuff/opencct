//! Module for probability distributions used in call center simulations.

use std::time::Duration;


/// Trait for probability distributions.
pub trait Distribution {
    /// Sample a value from the distribution at a given time.
    /// # Arguments
    /// * `at` - Duration since the start of the simulation.
    /// # Returns
    /// * `f64` - Sampled value from the distribution.
    fn sample(&mut self, at: Duration) -> f64;
}

pub mod uniform;
pub use uniform::Uniform;
