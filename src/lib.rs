//! Welcome to OpenCCT, a library for simulating and optimally staffing call centers.

/// The numeric type used throughout opencct.
///
/// Currently set to `f64` for performance and precision.
/// Can be changed later if needed (e.g., `f32` or `Decimal`).
#[cfg(feature = "f32")]
pub type Float = f32;

/// The numeric type used throughout opencct.
///
/// Currently set to `f64` for performance and precision.
/// Can be changed later if needed (e.g., `f32` or `Decimal`).
#[cfg(feature = "f64")]
pub type Float = f64;

pub mod test_utils;

pub mod distributions;

pub mod time;
pub use time::TimeUnit;

pub mod math;
