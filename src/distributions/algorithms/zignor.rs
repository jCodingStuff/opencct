//! # Zignor method
//! Doornik, J. A. (2005).
//! [An Improved Ziggurat Method to Generate Normal Random Samples](https://www.doornik.com/research/ziggurat.pdf).
//! University of Oxford.

use rand::Rng;
use once_cell::sync::Lazy;

use crate::Float;

/// Number of blocks
const ZIGNOR_C: usize = 128;
/// Start of the right tail
const ZIGNOR_R: Float = 3.442619855899;
/// (R * phi(R) + Pr(X>=R)) * sqrt(2\pi)
const ZIGNOR_V: Float = 9.91256303526217e-3;

/// Struct to hold the Zignor tables
/// `s_ad_zig_x` holds coordinates, such that each rectangle has same area;
/// `s_ad_zig_r` holds `s_ad_zig_x[i + 1] / s_ad_zig_x[i]`
struct ZignorTables {
    s_ad_zig_x: [Float; ZIGNOR_C + 1],
    s_ad_zig_r: [Float; ZIGNOR_C],
}

impl ZignorTables {
    fn build() -> Self {
        let (mut s_ad_zig_x, mut s_ad_zig_r) = ([0.0; ZIGNOR_C + 1], [0.0; ZIGNOR_C]);

        let mut f = (-0.5 * ZIGNOR_R * ZIGNOR_R).exp();
        s_ad_zig_x[0] = ZIGNOR_V / f;  // [0] is bottom block: V / f(R)
        s_ad_zig_x[1] = ZIGNOR_R;
        // s_ad_zig_x[ZIGNOR_C] = 0.0;  // Not needed since everything is initialized to 0

        for i in 2..ZIGNOR_C {
            s_ad_zig_x[i] = (-2.0 * (ZIGNOR_V / s_ad_zig_x[i-1] + f).ln()).sqrt();
            f = (-0.5 * s_ad_zig_x[i] * s_ad_zig_x[i]).exp();
        }
        for i in 0..ZIGNOR_C {
            s_ad_zig_r[i] = s_ad_zig_x[i+1] / s_ad_zig_x[i];
        }

        Self { s_ad_zig_x, s_ad_zig_r }
    }
}

/// Singleton for Zignor Tables
static ZIGNOR_TABLES: Lazy<ZignorTables> = Lazy::new(ZignorTables::build);

fn normal_tail<R: Rng + ?Sized>(rng: &mut R, min: Float, negative: bool) -> Float {
    let mut x = rng.random::<Float>().ln() / min;
    let mut y = rng.random::<Float>().ln();
    while -2.0 * y < x * x {
        x = rng.random::<Float>().ln() / min;
        y = rng.random::<Float>().ln();
    }
    return if negative { x - min } else { min - x };
}

/// Zignor Method
/// # Arguments
/// * `rng` - A random number generator
/// # Returns
/// A random variable that has the standard normal distribution
pub fn zignor_method<R: Rng + ?Sized>(rng: &mut R) -> Float {
    loop {
        let u = rng.random::<Float>() * 2.0 - 1.0;
        let i = rng.random::<u32>() as usize & 0x7F;
        let s_ad_zig_r_i = ZIGNOR_TABLES.s_ad_zig_r[i];
        let s_ad_zig_x_i = ZIGNOR_TABLES.s_ad_zig_x[i];

        // First try the rectangular boxes
        if u.abs() < s_ad_zig_r_i {
            return u * s_ad_zig_x_i;
        }

        // Bottom box: sample from the tail
        if i == 0 {
            return normal_tail(rng, ZIGNOR_R, u < 0.0);
        }

        let s_ad_zig_x_ip1 = ZIGNOR_TABLES.s_ad_zig_x[i+1];

        // Is this a sample from the wedges?
        let x = u * s_ad_zig_x_i;
        let f0 = (-0.5 * (s_ad_zig_x_i * s_ad_zig_x_i - x * x)).exp();
        let f1 = (-0.5 * (s_ad_zig_x_ip1 * s_ad_zig_x_ip1 - x * x)).exp();
        if f1 + rng.random::<Float>() * (f0 - f1) < 1.0 {
            return x;
        }
    }
}

/// Scaled Zignor Method
/// # Arguments
/// * `rng` - A random number generator
/// * `mu` - Mean of the desired normal distribution
/// * `sigma` - Standard deviation of the desired normal distribution
/// # Returns
/// A random variable that has the normal distribution with mean `mu` and standard deviation `sigma`
pub fn scaled_zignor_method<R: Rng + ?Sized>(rng: &mut R, mu: Float, sigma: Float) -> Float {
    mu + zignor_method(rng) * sigma
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    #[ignore]
    fn basic_statistics() {
        const N_SAMPLES: usize = 100_000;
        const TOLERANCE_PERCENT: Float = 1.0; // 1% tolerance

        let mu = 5.0;
        let sigma = 2.0;

        let mut rng = StdRng::from_seed([42; 32]);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..N_SAMPLES {
            let x = scaled_zignor_method(&mut rng, mu, sigma);
            sum += x;
            sum_sq += x * x;
        }

        let mean = sum / N_SAMPLES as Float;
        let variance = sum_sq / N_SAMPLES as Float - mean * mean;
        let std = variance.sqrt();

        let mean_tol = mu * TOLERANCE_PERCENT / 100.0;
        let std_tol = sigma * TOLERANCE_PERCENT / 100.0;

        assert!(
            (mean - mu).abs() <= mean_tol,
            "Mean too far from {mu}: {mean} (tolerance ±{mean_tol})",
        );
        assert!(
            (std - sigma).abs() <= std_tol,
            "Std too far from {sigma}: {std} (tolerance ±{std_tol})",
        );
    }

    #[test]
    fn reproducible_with_seed() {
        let seed = [123; 32];
        let mu = 1.0;
        let sigma = 3.0;

        let mut rng1 = StdRng::from_seed(seed);
        let mut rng2 = StdRng::from_seed(seed);

        for _ in 0..1000 {
            let val1 = scaled_zignor_method(&mut rng1, mu, sigma);
            let val2 = scaled_zignor_method(&mut rng2, mu, sigma);
            assert_eq!(val1, val2, "Values should be equal with same seed");
        }
    }

    #[test]
    #[ignore]
    fn values_are_finite() {
        let mut rng = StdRng::from_seed([99; 32]);
        let mu = 0.0;
        let sigma = 1.0;

        for _ in 0..10_000 {
            let x = scaled_zignor_method(&mut rng, mu, sigma);
            assert!(x.is_finite(), "Generated value is not finite: {x}");
        }
    }

    #[test]
    #[ignore]
    fn extreme_values() {
        let mut rng = StdRng::from_seed([7; 32]);
        let mu = 0.0;
        let sigma = 10.0;

        for _ in 0..10_000 {
            let _ = scaled_zignor_method(&mut rng, mu, sigma);
        }
    }
}
