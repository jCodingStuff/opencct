//! Algorithms for sampling from probability distributions

#[cfg(feature = "f32")]
use fast_inv_sqrt::InvSqrt32;

#[cfg(feature = "f64")]
use fast_inv_sqrt::InvSqrt64;

use crate::Float;

/// Inverse square root with [Quake III algorithm](https://en.wikipedia.org/wiki/Fast_inverse_square_root)
pub fn inv_sqrt(x: Float) -> Float {
    #[cfg(feature = "f32")]
    { x.inv_sqrt32() }

    #[cfg(feature = "f64")]
    { x.inv_sqrt64() }

}

pub mod box_muller;
pub use box_muller::{box_muller_transform, scaled_box_muller_transform};

pub mod zignor;
pub use zignor::{zignor_method, scaled_zignor_method};

pub mod marsaglia_tsang;
