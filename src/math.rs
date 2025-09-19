//! Math utilities

use super::Float;

/// Gamma function
pub fn gamma(x: Float) -> Float {
    #[cfg(feature = "f32")]
    { libm::tgammaf(x) }

    #[cfg(feature = "f64")]
    { libm::tgamma(x) }
}
