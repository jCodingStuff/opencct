//! Module for probability distributions used in call center simulations.

use std::time::Duration;
use rand::RngCore;

/// Trait for probability distributions.
/// All structs implementing this trait must know that the base unit is seconds.
pub trait Distribution {
    /// Get the theoretical mean of the distribution at a given time.
    /// Distributions that care about theoretical statistics should override this method.
    /// The default implementation returns [Duration::ZERO].
    /// # Arguments
    /// * `at` - the requested time as [Duration]
    /// # Returns
    /// The theoretical mean as [Duration]
    fn mean(&self, _at: Duration) -> Duration { Duration::ZERO }

    /// Get the theoretical mean of the distribution at time 0.
    /// # Returns
    /// The theoretical mean as [Duration]
    fn mean_at_t0(&self) -> Duration {
        self.mean(Duration::ZERO)
    }

    /// Get the theoretical variance of the distribution at a given time.
    /// Distributions that care about theoretical statistics should override this method.
    /// The default implementation returns [Duration::ZERO].
    /// # Arguments
    /// * `at` - the requested time as [Duration]
    /// # Returns
    /// The theoretical variance as [Duration]
    fn variance(&self, _at: Duration) -> Duration { Duration::ZERO }

    /// Get the theoretical variance of the distribution at time 0.
    /// # Returns
    /// The theoretical variance as [Duration]
    fn variance_at_t0(&self) -> Duration {
        self.variance(Duration::ZERO)
    }

    /// Sample a value from the distribution at a given time.
    /// # Arguments
    /// * `at` - [Duration] since the start of the simulation.
    /// * `rng` - Random number generator
    /// # Returns
    /// * [Duration] - Sampled value from the distribution.
    fn sample(&self, at: Duration, rng: &mut dyn RngCore) -> Duration;

    /// Sample a value from the distribution at time 0
    /// # Arguments
    /// * `rng` - Random number generator
    /// # Returns
    /// [Duration] - Sampled value from the distribution.
    fn sample_at_t0(&self, rng: &mut dyn RngCore) -> Duration {
        self.sample(Duration::ZERO, rng)
    }

    /// Sample `n` values from the distribution at a given time.
    /// # Arguments
    /// * `n` - Amount of values
    /// * `at` - [Duration] since the start of the simulation.
    /// * `rng` - Random number generator
    /// # Returns
    /// * A vector of [Duration] - Sampled values from the distribution.
    fn sample_n(&self, n: usize, at: Duration, rng: &mut dyn RngCore) -> Vec<Duration> {
        (0..n).map(|_| self.sample(at, rng)).collect()
    }

    /// Sample `n` values from the distribution at time 0.
    /// # Arguments
    /// * `n` - Amount of values
    /// * `rng` - Random number generator
    /// # Returns
    /// * A vector of [Duration] - Sampled values from the distribution.
    fn sample_n_at_t0(&self, n: usize, rng: &mut dyn RngCore) -> Vec<Duration> {
        self.sample_n(n, Duration::ZERO, rng)
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

pub mod weibull;
pub use weibull::Weibull;
pub use weibull::WeibullTV;

pub mod triangular;
pub use triangular::Triangular;
pub use triangular::TriangularTV;

pub mod pareto;
pub use pareto::Pareto;
pub use pareto::ParetoTV;

pub mod beta;
pub use beta::Beta;
pub use beta::BetaTV;

pub mod composite;
pub use composite::CompositeEntry;
pub use composite::Composite;
