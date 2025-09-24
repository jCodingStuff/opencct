//! Composite distribution

use std::time::Duration;
use rand::RngCore;

use super::Distribution;

/// A single entry in a [Composite] distribution.
/// Associates a lower bound in simulation time with a probability distribution.
pub struct CompositeEntry {
    /// The lower bound time at which this distribution becomes active.
    pub lower_bound     : Duration,
    /// The distribution active starting at this lower bound.
    pub distribution    : Box<dyn Distribution>,
}

/// A composite distribution.
///
/// The [Composite] distribution allows combining multiple probability
/// distributions into one, with each distribution being active only after
/// a specified lower bound of simulation time. This is useful when different
/// time intervals in a simulation should follow different statistical behaviors.
///
/// At exactly `lower_bound`, that distribution becomes active.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use rand::{rngs::StdRng, SeedableRng};
/// use opencct::distributions::{Distribution, Composite, CompositeEntry, Uniform, Triangular, Exponential};
/// use opencct::time::TimeUnit;
///
/// // Create a composite distribution:
/// // - Uniform between 1–2s until t=10
/// // - Triangular between 2–5s with mode 3s until t=20
/// // - Exponential with mean 1s after t=20
/// let composite = Composite::new(vec![
///     CompositeEntry {
///         lower_bound: Duration::from_secs(0),
///         distribution: Box::new(Uniform::new(1.0, 2.0, TimeUnit::Seconds)),
///     },
///     CompositeEntry {
///         lower_bound: Duration::from_secs(10),
///         distribution: Box::new(Triangular::new(2.0, 5.0, 3.0, TimeUnit::Seconds)),
///     },
///     CompositeEntry {
///         lower_bound: Duration::from_secs(20),
///         distribution: Box::new(Exponential::new(1.0, TimeUnit::Seconds)),
///     },
/// ]);
///
/// let mut rng = StdRng::seed_from_u64(42);
///
/// // Sample at different times
/// let sample_5  = composite.sample(Duration::from_secs(5),  &mut rng);
/// let sample_15 = composite.sample(Duration::from_secs(15), &mut rng);
/// let sample_25 = composite.sample(Duration::from_secs(25), &mut rng);
///
/// println!("Sample at 5s:  {:?}", sample_5);
/// println!("Sample at 15s: {:?}", sample_15);
/// println!("Sample at 25s: {:?}", sample_25);
/// ```
///
/// # Panics
/// Panics if:
/// - `distributions` is empty
/// - `distributions` is not sorted by strictly ascending `lower_bound`
pub struct Composite {
    distributions: Vec<CompositeEntry>,
}

impl Composite {
    /// Create a new [Composite] distribution.
    /// # Arguments
    /// * `distributions` - The distributions and time bounds that make up the composite
    /// # Returns
    /// A new [Composite]
    /// # Panic
    /// Panics if:
    /// - `distributions` is empty
    /// - `distributions` is not sorted by strictly ascending `lower_bound`
    pub fn new(distributions: Vec<CompositeEntry>) -> Self {
        assert!(!distributions.is_empty(), "There are no distributions in the slice!");
        assert!(
            distributions.windows(2).all(|w| w[0].lower_bound < w[1].lower_bound),
            "The distributions slice is not sorted by ascending lower_bound or a given lower_bound is duplicated",
        );
        Self { distributions }
    }
}

impl Distribution for Composite {
    fn sample(&self, at: Duration, rng: &mut dyn RngCore) -> Duration {
        let idx = match self.distributions.binary_search_by(
            |entry| entry.lower_bound.cmp(&at)
        ) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        self.distributions[idx].distribution.sample(at, rng)
    }
}

#[cfg(test)]
mod composite_tests {
    use super::*;
    use crate::distributions::{
        uniform::Uniform,
        triangular::Triangular,
        exponential::Exponential,
    };
    use crate::time::TimeUnit;
    use rand::{rngs::StdRng, SeedableRng};

    fn create_test_composite() -> Composite {
        let distributions = vec![
            CompositeEntry {
                lower_bound: Duration::from_secs(0),
                distribution: Box::new(Uniform::new(1.0, 2.0, TimeUnit::Seconds)),
            },
            CompositeEntry {
                lower_bound: Duration::from_secs(10),
                distribution: Box::new(Triangular::new(2.0, 5.0, 3.0, TimeUnit::Millis)),
            },
            CompositeEntry {
                lower_bound: Duration::from_secs(20),
                distribution: Box::new(Exponential::new(1.0, TimeUnit::Minutes)),
            },
        ];

        Composite::new(distributions)
    }

    #[test]
    #[should_panic]
    fn panics_on_empty_distributions() {
        let _ = Composite::new(vec![]);
    }

    #[test]
    #[should_panic]
    fn constructor_panics_on_unsorted_lower_bounds() {
        let _ = Composite::new(vec![
            CompositeEntry { lower_bound: Duration::from_secs(10), distribution: Box::new(Uniform::new(1.0, 2.0, TimeUnit::Seconds)) },
            CompositeEntry { lower_bound: Duration::from_secs(5), distribution: Box::new(Uniform::new(1.0, 2.0, TimeUnit::Seconds)) },
        ]);
    }
}
