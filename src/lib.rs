//! Welcome to OpenCCT, a library for simulating and optimally staffing call centers.

use std::time::Duration;

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

    /// Convert the duration to minutes as a floating-point number.
    /// # Returns
    /// * `f64` - Duration in minutes.
    fn as_minutes_f64(&self) -> f64;

    /// Convert the duration to hours as a floating-point number.
    /// # Returns
    /// * `f64` - Duration in hours.
    fn as_hours_f64(&self) -> f64;
}

impl DurationExtension for Duration {
    fn from_minutes(minutes: u64) -> Duration {
        Duration::from_secs(minutes * 60)
    }

    fn from_hours(hours: u64) -> Duration {
        Duration::from_secs(hours * 3600)
    }

    fn as_minutes_f64(&self) -> f64 {
        self.as_secs_f64() / 60.0
    }

    fn as_hours_f64(&self) -> f64 {
        self.as_secs_f64() / 3600.0
    }
}

pub mod distributions;
