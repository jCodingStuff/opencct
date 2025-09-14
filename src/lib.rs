//! Welcome to OpenCCT, a library for simulating and optimally staffing call centers.

use std::time::Duration;

/// The numeric type used throughout opencct.
///
/// Currently set to `f64` for performance and precision.
/// Can be changed later if needed (e.g., `f32` or `Decimal`).
#[cfg(feature = "f32")]
pub type Float = f32;

#[cfg(feature = "f64")]
pub type Float = f64;

/// Extension trait for `Duration` to provide additional time unit conversions.
/// Bring it into scope with `use opencct::DurationExt` in order to use its methods.
pub trait DurationExtension {
    /// Construct a new Duration from minutes
    /// # Arguments
    /// * `minutes` - The amount of minutes
    /// # Returns
    /// * `Duration` - A new instance of Duration
    fn from_minutes(minutes: u64) -> Duration;

    /// Construct a new Duration from hours
    /// # Arguments
    /// * `hours` - The amount of hours
    /// # Returns
    /// * `Duration` - A new instance of Duration
    fn from_hours(hours: u64) -> Duration;

    /// Convert the duration to seconds as a floating-point number.
    /// # Returns
    /// * [Float] - Duration in seconds.
    fn as_secs_float(&self) -> Float;

    /// Convert the duration to minutes as a floating-point number.
    /// # Returns
    /// * [Float] - Duration in minutes.
    fn as_minutes_float(&self) -> Float;

    /// Convert the duration to hours as a floating-point number.
    /// # Returns
    /// * [Float] - Duration in hours.
    fn as_hours_float(&self) -> Float;
}

impl DurationExtension for Duration {
    fn from_minutes(minutes: u64) -> Duration {
        Duration::from_secs(minutes * 60)
    }

    fn from_hours(hours: u64) -> Duration {
        Duration::from_secs(hours * 3600)
    }

    fn as_secs_float(&self) -> Float {
        #[cfg(feature = "f32")]
        self.as_secs_f32();

        #[cfg(feature = "f64")]
        self.as_secs_f64()
    }

    fn as_minutes_float(&self) -> Float {
        self.as_secs_float() / 60.0
    }

    fn as_hours_float(&self) -> Float {
        self.as_secs_float() / 3600.0
    }
}

pub mod distributions;
