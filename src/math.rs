//! Math utilities

use super::Float;

/// Gamma function
pub fn gamma(x: Float) -> Float {
    #[cfg(all(feature = "f32", not(feature = "f64")))]
    {
        libm::tgammaf(x)
    }

    #[cfg(feature = "f64")]
    {
        libm::tgamma(x)
    }
}
