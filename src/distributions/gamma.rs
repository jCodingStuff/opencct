//! Gamma-Erlang distribution
//!
//! Erlang is just a Gamma with an integer shape parameter

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

/// Gamma-Erlang distribution.
///
/// Implemented via the Marsaglia-Tsang method. See
/// Marsaglia, G., & Tsang, W. W. (2000).
/// [A simple method for generating gamma variables](https://doi.acm.org/10.1145/358407.358414).
/// *ACM Transactions on Mathematical Software (TOMS)*, 26(3), 363-372.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, GammaErlang};
/// use opencct::time::TimeUnit;
///
/// let mut dist = GammaErlang::new(1.0, 3.0, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct GammaErlang {
    /// Scale parameter
    theta   : Float,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
    /// Sampling method struct
    method  : MarsagliaTsang,
}

impl GammaErlang {
    /// Create a new [GammaErlang] distribution with given shape and scale values.
    /// # Arguments
    /// * `alpha` - Shape
    /// * `theta` - Scale
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [GammaErlang].
    /// # Panic
    /// This function panics if either `alpha` or `theta` are <= 0
    pub fn new(alpha: Float, theta: Float, unit: TimeUnit) -> Self {
        assert!(alpha > 0.0 && theta > 0.0, "Invalid alpha {alpha} or theta {theta}");
        Self {
            theta,
            factor  : unit.factor(),
            rng     : StdRng::from_os_rng(),
            method  : MarsagliaTsang::setup(alpha),
        }
    }

    /// Create a new [GammaErlang] distribution with given random seed
    /// # Arguments
    /// * `alpha` - Shape
    /// * `theta` - Scale
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// * A new [GammaErlang].
    /// # Panic
    /// This function panics if either `alpha` or `theta` are <= 0
    pub fn new_seeded(alpha: Float, theta: Float, unit: TimeUnit, seed: u64) -> Self {
        assert!(alpha > 0.0 && theta > 0.0, "Invalid alpha {alpha} or theta {theta}");
        Self {
            theta,
            factor  : unit.factor(),
            rng     : StdRng::seed_from_u64(seed),
            method  : MarsagliaTsang::setup(alpha),
        }
    }
}

impl Distribution for GammaErlang {
    fn sample(&mut self, _: Duration) -> Duration {
        let raw = self.method.sample_from_setup(&mut self.rng, self.theta);
        Duration::from_secs_float(raw * self.factor)
    }
}

/// Gamma-Erlang distribution with time-varying shape and scale.
///
/// Implemented via the Marsaglia-Tsang method. See
/// Marsaglia, G., & Tsang, W. W. (2000).
/// [A simple method for generating gamma variables](https://doi.acm.org/10.1145/358407.358414).
/// *ACM Transactions on Mathematical Software (TOMS)*, 26(3), 363-372.
///
/// # Example
/// ```
/// use std::time::Duration;
/// use opencct::distributions::{Distribution, GammaErlangTV};
/// use opencct::time::{TimeUnit, DurationExtension};
///
/// let mut dist = GammaErlangTV::new(|t| 1.0 + t.as_secs_float() * 0.1, |t| 3.0 + t.as_secs_float() * 0.1, TimeUnit::Seconds);
/// let sample = dist.sample_at_t0();
/// println!("Sampled value: {:?}", sample);
/// ```
pub struct GammaErlangTV<Fa, Fb> {
    /// Shape parameter as a function of time
    alpha   : Fa,
    /// Scale parameter as a function of time
    theta   : Fb,
    /// Time unit factor
    factor  : Float,
    /// Random number generator
    rng     : StdRng,
}

impl<Fa, Fb> GammaErlangTV<Fa, Fb>
where
    Fa: Fn(Duration) -> Float,
    Fb: Fn(Duration) -> Float,
{
    /// Create a new [GammaErlang] distribution with given shape and scale functions.
    /// # Arguments
    /// * `alpha` - Function to compute the shape at a given time. Must be > 0 for any t >= 0
    /// * `theta` - Function to compute the scale at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// # Returns
    /// * A new [GammaErlang].
    /// # Be careful!
    /// `alpha` and `theta` values are not checked in release mode! Make sure you fulfill the contract!
    pub fn new(alpha: Fa, theta: Fb, unit: TimeUnit) -> Self {
        Self {
            alpha,
            theta,
            factor  : unit.factor(),
            rng     : StdRng::from_os_rng(),
        }
    }

    /// Create a new [GammaErlang] distribution with given random seed
    /// # Arguments
    /// * `alpha` - Function to compute the shape at a given time. Must be > 0 for any t >= 0
    /// * `theta` - Function to compute the scale at a given time. Must be > 0 for any t >= 0
    /// * `unit` - The [TimeUnit] that the distribution samples.
    /// * `seed` - Seed for the random number generator.
    /// # Returns
    /// * A new [GammaErlang].
    /// # Be careful!
    /// `alpha` and `theta` values are not checked in release mode! Make sure you fulfill the contract!
    pub fn new_seeded(alpha: Fa, theta: Fb, unit: TimeUnit, seed: u64) -> Self {
        Self {
            alpha,
            theta,
            factor  : unit.factor(),
            rng     : StdRng::seed_from_u64(seed),
        }
    }
}

impl<Fa, Fb> Distribution for GammaErlangTV<Fa, Fb>
where
    Fa: Fn(Duration) -> Float,
    Fb: Fn(Duration) -> Float,
{
    /// See [Distribution::sample]
    /// # Panic
    /// In debug, this function will panic if at the requested time the shape or scale are <= 0.
    /// **This is NOT checked in release mode!**
    fn sample(&mut self, at: Duration) -> Duration {
        let alpha = (self.alpha)(at);
        let theta = (self.theta)(at);
        debug_assert!(alpha > 0.0 && theta > 0.0, "Invalid alpha {alpha} or {theta} bound at {at:?}");
        let raw = MarsagliaTsang::sample(&mut self.rng, alpha, theta);
        Duration::from_secs_float(raw * self.factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn new_panics_on_zero_alpha() {
        let _ = GammaErlang::new(0.0, 1.0, TimeUnit::Seconds);
    }

    #[test]
    #[should_panic]
    fn new_panics_on_zero_theta() {
        let _ = GammaErlang::new(1.0, 0.0, TimeUnit::Seconds);
    }

    #[test]
    #[should_panic]
    fn new_seeded_panics_on_negative_alpha() {
        let _ = GammaErlang::new_seeded(-1.0, 1.0, TimeUnit::Seconds, 42);
    }

    #[test]
    #[should_panic]
    fn new_seeded_panics_on_negative_theta() {
        let _ = GammaErlang::new_seeded(1.0, -1.0, TimeUnit::Seconds, 42);
    }

    #[test]
    fn smoke_sample() {
        let mut dist = GammaErlang::new(2.0, 3.0, TimeUnit::Seconds);
        let _ = dist.sample_at_t0();
    }

    #[test]
    fn reproducible_with_seed() {
        let seed = 42;
        let mut dist1 = GammaErlang::new_seeded(2.5, 1.0, TimeUnit::Seconds, seed);
        let mut dist2 = GammaErlang::new_seeded(2.5, 1.0, TimeUnit::Seconds, seed);

        for _ in 0..1000 {
            let val1 = dist1.sample_at_t0();
            let val2 = dist2.sample_at_t0();
            assert_eq!(val1, val2, "Values should be equal with same seed");
        }
    }

    #[test]
    #[ignore]
    fn mean_and_variance_large_sample() {
        const N_SAMPLES: usize = 100_000;
        const TOLERANCE_PERCENT: Float = 1.0;

        let alpha = 2.0;
        let theta = 3.0;

        let mut dist = GammaErlang::new(alpha, theta, TimeUnit::Seconds);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..N_SAMPLES {
            let x = dist.sample_at_t0().as_secs_float();
            sum += x;
            sum_sq += x * x;
        }

        let mean = sum / N_SAMPLES as Float;
        let variance = sum_sq / N_SAMPLES as Float - mean * mean;
        let std = variance.sqrt();

        let expected_mean = alpha * theta;
        let expected_std = (alpha * theta * theta).sqrt();

        let mean_tol = expected_mean * TOLERANCE_PERCENT / 100.0;
        let std_tol = expected_std * TOLERANCE_PERCENT / 100.0;

        assert!(
            (mean - expected_mean).abs() <= mean_tol,
            "Mean too far from {expected_mean}: {mean} (tolerance ±{mean_tol})"
        );
        assert!(
            (std - expected_std).abs() <= std_tol,
            "Std too far from {expected_std}: {std} (tolerance ±{std_tol})"
        );
    }

    #[test]
    fn values_are_finite() {
        let mut dist = GammaErlang::new(3.0, 2.0, TimeUnit::Seconds);

        for _ in 0..10_000 {
            let x = dist.sample_at_t0().as_secs_float();
            assert!(x.is_finite(), "Generated value is not finite: {x}");
        }
    }
}

#[cfg(test)]
mod tests_tv {
    use super::*;

    #[test]
    fn smoke_sample_tv() {
        let mut dist = GammaErlangTV::new(|_| 2.0, |_| 3.0, TimeUnit::Seconds);
        let _ = dist.sample_at_t0();
        let _ = dist.sample(Duration::from_secs(5));
    }

    #[test]
    fn reproducible_with_seed_tv() {
        let seed = 99;
        let mut dist1 = GammaErlangTV::new_seeded(|_| 2.5, |_| 1.0, TimeUnit::Seconds, seed);
        let mut dist2 = GammaErlangTV::new_seeded(|_| 2.5, |_| 1.0, TimeUnit::Seconds, seed);

        for _ in 0..1000 {
            let val1 = dist1.sample_at_t0();
            let val2 = dist2.sample_at_t0();
            assert_eq!(val1, val2, "Values should be equal with same seed");
        }
    }

    #[test]
    #[ignore]
    fn mean_and_variance_large_sample_tv() {
        const N_SAMPLES: usize = 100_000;
        const TOLERANCE_PERCENT: Float = 1.0;

        let alpha = 2.0;
        let theta = 3.0;

        let mut dist = GammaErlangTV::new(|_| alpha, |_| theta, TimeUnit::Seconds);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..N_SAMPLES {
            let x = dist.sample_at_t0().as_secs_float();
            sum += x;
            sum_sq += x * x;
        }

        let mean = sum / N_SAMPLES as Float;
        let variance = sum_sq / N_SAMPLES as Float - mean * mean;
        let std = variance.sqrt();

        let expected_mean = alpha * theta;
        let expected_std = (alpha * theta * theta).sqrt();

        let mean_tol = expected_mean * TOLERANCE_PERCENT / 100.0;
        let std_tol = expected_std * TOLERANCE_PERCENT / 100.0;

        assert!(
            (mean - expected_mean).abs() <= mean_tol,
            "Mean too far from {expected_mean}: {mean} (tolerance ±{mean_tol})"
        );
        assert!(
            (std - expected_std).abs() <= std_tol,
            "Std too far from {expected_std}: {std} (tolerance ±{std_tol})"
        );
    }
}
