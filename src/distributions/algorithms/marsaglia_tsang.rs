//! # Marsaglia-Tsang Method for generating Gamma variables
//! Marsaglia, G., & Tsang, W. W. (2000).
//! [A simple method for generating gamma variables](https://doi.acm.org/10.1145/358407.358414).
//! *ACM Transactions on Mathematical Software (TOMS)*, 26(3), 363-372.

use rand::Rng;

use crate::Float;
use super::{
    zignor::zignor_method,
    inv_sqrt,
};

/// Marsaglia-Tsang Method struct
#[derive(Debug, Copy, Clone)]
pub struct MarsagliaTsang {
    alpha   : Float,
    d       : Float,
    c       : Float,
}

impl MarsagliaTsang {
    /// Setup the Marsaglia-Tsang Method
    /// # Arguments
    /// * `alpha` - Shape
    /// # Returns
    /// A new [MarsagliaTsang]
    pub fn setup(alpha: Float) -> Self {
        let d = if alpha < 1.0 { 1.0 } else { 0.0 } + alpha - 1.0 / 3.0;
        Self { alpha, d, c: inv_sqrt(9.0*d) }
    }

    /// Perform the Marsaglia-Tsang Method from an existing [MarsagliaTsang]
    /// setup (see [MarsagliaTsang::setup])
    /// # Arguments
    /// * `rng` - A random number generator
    /// * `theta` - Scale
    /// # Returns
    /// A random variable that has a gamma distribution with shape `alpha` and scale `theta`
    pub fn sample_from_setup<R: Rng + ?Sized>(&self, rng: &mut R, theta: Float) -> Float {
        let (mut x, mut v, mut u);
        loop {
            x = zignor_method(rng);
            v = 1.0 + self.c * x;
            while v <= 0.0 {
                x = zignor_method(rng);
                v = 1.0 + self.c * x;
            }
            v = v * v * v;
            u = rng.random::<Float>();

            if u < 1.0 - 0.0331 * x * x * x * x || u.ln() < 0.5 * x * x + self.d * (1.0 - v + v.ln()) {
                return theta * self.d * v
                    * if self.alpha < 1.0 { rng.random::<Float>().powf(1.0 / self.alpha) } else { 1.0 };
            }
        }
    }

    /// Perform the Marsaglia-Tsang Method
    ///
    /// For performance with a fixed shape `alpha`, use [MarsagliaTsang::setup] and [MarsagliaTsang::sample_from_setup]
    /// # Arguments
    /// * `rng` - A random number generator
    /// * `alpha` - Shape
    /// * `theta` - Scale
    /// # Returns
    /// A random variable that has a gamma distribution with shape `alpha` and scale `theta`
    pub fn sample<R: Rng + ?Sized>(rng: &mut R, alpha: Float, theta: Float) -> Float {
        Self::setup(alpha).sample_from_setup(rng, theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn smoke_test_gamma() {
        let mut rng = StdRng::from_seed([1; 32]);
        let alpha = 2.0;
        let theta = 3.0;

        for _ in 0..100 {
            let x = MarsagliaTsang::sample(&mut rng, alpha, theta);
            assert!(x.is_finite(), "Generated value not finite: {x}");
            assert!(x >= 0.0, "Gamma variates must be non-negative, got {x}");
        }
    }

    #[test]
    fn reproducibility() {
        let seed = [7; 32];
        let alpha = 2.5;
        let theta = 1.0;

        let mut rng1 = StdRng::from_seed(seed);
        let mut rng2 = StdRng::from_seed(seed);

        for _ in 0..100 {
            let x1 = MarsagliaTsang::sample(&mut rng1, alpha, theta);
            let x2 = MarsagliaTsang::sample(&mut rng2, alpha, theta);
            assert_eq!(x1, x2, "Values must be identical with same seed");
        }
    }

    #[test]
    #[ignore] // statistical test, not for every CI run
    fn mean_and_variance_are_close() {
        const N_SAMPLES: usize = 200_000;
        const TOLERANCE_PERCENT: Float = 3.0;

        let alpha = 5.0;
        let theta = 2.0;
        let expected_mean = alpha * theta;
        let expected_var = alpha * theta * theta;

        let mut rng = StdRng::from_seed([42; 32]);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..N_SAMPLES {
            let x = MarsagliaTsang::sample(&mut rng, alpha, theta);
            sum += x;
            sum_sq += x * x;
        }

        let mean = sum / N_SAMPLES as Float;
        let var = sum_sq / N_SAMPLES as Float - mean * mean;

        let mean_tol = expected_mean * TOLERANCE_PERCENT / 100.0;
        let var_tol = expected_var * TOLERANCE_PERCENT / 100.0;

        assert!(
            (mean - expected_mean).abs() <= mean_tol,
            "Sample mean {mean} differs from expected {expected_mean} (tolerance ±{mean_tol})",
        );
        assert!(
            (var - expected_var).abs() <= var_tol,
            "Sample variance {var} differs from expected {expected_var} (tolerance ±{var_tol})",
        );
    }

    #[test]
    #[ignore] // statistical test, alpha < 1 branch
    fn mean_and_variance_with_alpha_less_than_one() {
        const N_SAMPLES: usize = 200_000;
        const TOLERANCE_PERCENT: Float = 5.0;

        let alpha = 0.7;
        let theta = 2.0;
        let expected_mean = alpha * theta;
        let expected_var = alpha * theta * theta;

        let mut rng = StdRng::from_seed([77; 32]);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..N_SAMPLES {
            let x = MarsagliaTsang::sample(&mut rng, alpha, theta);
            assert!(x >= 0.0, "Gamma variates must be non-negative, got {x}");
            sum += x;
            sum_sq += x * x;
        }

        let mean = sum / N_SAMPLES as Float;
        let var = sum_sq / N_SAMPLES as Float - mean * mean;

        let mean_tol = expected_mean * TOLERANCE_PERCENT / 100.0;
        let var_tol = expected_var * TOLERANCE_PERCENT / 100.0;

        assert!(
            (mean - expected_mean).abs() <= mean_tol,
            "Sample mean {mean} differs from expected {expected_mean} (tolerance ±{mean_tol})",
        );
        assert!(
            (var - expected_var).abs() <= var_tol,
            "Sample variance {var} differs from expected {expected_var} (tolerance ±{var_tol})",
        );
    }

    #[test]
    fn sample_and_sample_from_setup_are_consistent() {
        let alpha = 3.0;
        let theta = 2.0;
        let setup = MarsagliaTsang::setup(alpha);

        let seed = [11; 32];
        let mut rng1 = StdRng::from_seed(seed);
        let mut rng2 = StdRng::from_seed(seed);

        for _ in 0..100 {
            let x1 = MarsagliaTsang::sample(&mut rng1, alpha, theta);
            let x2 = setup.sample_from_setup(&mut rng2, theta);
            assert_eq!(x1, x2, "sample and sample_from_setup should produce identical values with same seed");
        }
    }
}
