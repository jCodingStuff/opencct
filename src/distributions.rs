//! Module for probability distributions used in call center simulations.

use std::time::Duration;

use crate::Float;


/// Trait for probability distributions.
pub trait Distribution {
    /// Sample a value from the distribution at a given time.
    /// # Arguments
    /// * `at` - [Duration] since the start of the simulation.
    /// # Returns
    /// * [Float] - Sampled value from the distribution.
    fn sample(&mut self, at: Duration) -> Float;

    /// Sample a value from the distribution at time 0
    /// # Returns
    /// [Float] - Sampled value from the distribution.
    fn sample_at_t0(&mut self) -> Float {
        self.sample(Duration::ZERO)
    }
}

pub mod uniform;
pub use uniform::UniformTV;
pub use uniform::Uniform;
pub mod exponential;
// pub use exponential::Exponential;
