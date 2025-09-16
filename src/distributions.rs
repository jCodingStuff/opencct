//! Module for probability distributions used in call center simulations.

use std::time::Duration;

/// Trait for probability distributions.
/// All structs implementing this trait must know that the base unit is seconds.
pub trait Distribution {
    /// Sample a value from the distribution at a given time.
    /// # Arguments
    /// * `at` - [Duration] since the start of the simulation.
    /// # Returns
    /// * [Duration] - Sampled value from the distribution.
    fn sample(&mut self, at: Duration) -> Duration;

    /// Sample a value from the distribution at time 0
    /// # Returns
    /// [Duration] - Sampled value from the distribution.
    fn sample_at_t0(&mut self) -> Duration {
        self.sample(Duration::ZERO)
    }
}

pub mod algorithms;

pub mod uniform;
pub use uniform::Uniform;
pub use uniform::UniformTV;

pub mod exponential;
pub use exponential::Exponential;
pub use exponential::ExponentialTV;

pub mod normal;
pub use normal::Normal;
pub use normal::NormalTV;

pub mod lognormal;
pub use lognormal::LogNormal;
pub use lognormal::LogNormalTV;

pub mod gamma;
pub use gamma::GammaErlang;
pub use gamma::GammaErlangTV;
