//! Utilitie and extensions for time

use std::time::Duration;

use crate::Float;

/// Represents a unit for time
#[derive(Debug, Copy, Clone)]
pub enum TimeUnit {
    Days,
    Hours,
    Minutes,
    Seconds,
    Millis,
    Nanos,
}

impl TimeUnit {
    /// Get the unit factor. The base is seconds.
    /// # Returns
    /// The unit factor as a [Float]
    pub fn factor(&self) -> Float {
        match self {
            TimeUnit::Days      => 86400.0,
            TimeUnit::Hours     => 3600.0,
            TimeUnit::Minutes   => 60.0,
            TimeUnit::Seconds   => 1.0,
            TimeUnit::Millis    => 1e-3,
            TimeUnit::Nanos     => 1e-9,
        }
    }

    /// Convert a value to a [Duration]
    /// # Arguments
    /// * `value` - The value
    /// # Returns
    /// A new [Duration]
    pub fn to_duration(&self, value: Float) -> Duration {
        <Duration as DurationExtension>::from_secs_float(value * self.factor())
    }
}

/// Extension trait for `Duration` to provide additional time unit conversions.
/// Bring it into scope in order to use its methods.
pub trait DurationExtension {
    /// Construct a new Duration from floating days
    /// # Arguments
    /// * `days` - The amount of days
    /// # Returns
    /// * `Duration` - A new instance of Duration
    fn from_days_float(days: Float) -> Duration;

    /// Construct a new Duration from floating hours
    /// # Arguments
    /// * `hours` - The amount of hours
    /// # Returns
    /// * `Duration` - A new instance of Duration
    fn from_hours_float(hours: Float) -> Duration;

    /// Construct a new Duration from floating minutes
    /// # Arguments
    /// * `minutes` - The amount of minutes
    /// # Returns
    /// * `Duration` - A new instance of Duration
    fn from_minutes_float(minutes: Float) -> Duration;

    /// Construct a new Duration from floating seconds
    /// # Arguments
    /// * `seconds` - The amount of seconds
    /// # Returns
    /// * `Duration` - A new instance of Duration
    fn from_secs_float(seconds: Float) -> Duration;

    /// Construct a new Duration from floating millis
    /// # Arguments
    /// * `millis` - The amount of millis
    /// # Returns
    /// * `Duration` - A new instance of Duration
    fn from_millis_float(millis: Float) -> Duration;

    /// Construct a new Duration from floating nanos
    /// # Arguments
    /// * `nanos` - The amount of nanos
    /// # Returns
    /// * `Duration` - A new instance of Duration
    fn from_nanos_float(nanos: Float) -> Duration;

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

    /// Convert the duration to nanos as a floating-point number.
    /// # Returns
    /// * [Float] - Duration in nanos.
    fn as_nanos_float(&self) -> Float;

    /// Convert the duration to millis as a floating-point number.
    /// # Returns
    /// * [Float] - Duration in millis.
    fn as_millis_float(&self) -> Float;

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

    /// Convert the duration to days as a floating-point number.
    /// # Returns
    /// * [Float] - Duration in days.
    fn as_days_float(&self) -> Float;
}

impl DurationExtension for Duration {
    fn from_days_float(days: Float) -> Duration {
        <Duration as DurationExtension>::from_secs_float(days * 86400.0)
    }

    fn from_hours_float(hours: Float) -> Duration {
        <Duration as DurationExtension>::from_secs_float(hours * 3600.0)
    }

    fn from_minutes_float(minutes: Float) -> Duration {
        <Duration as DurationExtension>::from_secs_float(minutes * 60.0)
    }

    fn from_secs_float(seconds: Float) -> Duration {
        #[cfg(feature = "f32")]
        { Duration::from_secs_f32(seconds) }

        #[cfg(feature = "f64")]
        { Duration::from_secs_f64(seconds) }
    }

    fn from_millis_float(millis: Float) -> Duration {
        <Duration as DurationExtension>::from_secs_float(millis * 1e-3)
    }

    fn from_nanos_float(nanos: Float) -> Duration {
        <Duration as DurationExtension>::from_secs_float(nanos * 1e-9)
    }

    fn from_minutes(minutes: u64) -> Duration {
        Duration::from_secs(minutes * 60)
    }

    fn from_hours(hours: u64) -> Duration {
        Duration::from_secs(hours * 3600)
    }

    fn as_nanos_float(&self) -> Float {
        self.as_secs_float() * 1e9
    }

    fn as_millis_float(&self) -> Float {
        self.as_secs_float() * 1e3
    }

    fn as_secs_float(&self) -> Float {
        #[cfg(feature = "f32")]
        { self.as_secs_f32() }

        #[cfg(feature = "f64")]
        { self.as_secs_f64() }
    }

    fn as_minutes_float(&self) -> Float {
        self.as_secs_float() / 60.0
    }

    fn as_hours_float(&self) -> Float {
        self.as_secs_float() / 3600.0
    }

    fn as_days_float(&self) -> Float {
        self.as_secs_float() / 86400.0
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Float, b: Float, epsilon: Float) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_timeunit_factor() {
        assert!(approx_eq(TimeUnit::Days.factor(), 86400.0, 1e-9));
        assert!(approx_eq(TimeUnit::Hours.factor(), 3600.0, 1e-9));
        assert!(approx_eq(TimeUnit::Minutes.factor(), 60.0, 1e-9));
        assert!(approx_eq(TimeUnit::Seconds.factor(), 1.0, 1e-9));
        assert!(approx_eq(TimeUnit::Millis.factor(), 1e-3, 1e-12));
        assert!(approx_eq(TimeUnit::Nanos.factor(), 1e-9, 1e-12));
    }

    #[test]
    fn test_timeunit_to_duration() {
        let one_min = TimeUnit::Minutes.to_duration(1.0);
        assert_eq!(one_min.as_secs(), 60);

        let half_hour = TimeUnit::Hours.to_duration(0.5);
        assert_eq!(half_hour.as_secs(), 1800);

        let millis = TimeUnit::Millis.to_duration(250.0);
        assert_eq!(millis.as_millis(), 250);

        let nanos = TimeUnit::Nanos.to_duration(1_000_000.0);
        assert_eq!(nanos.as_millis(), 1);
    }

    #[test]
    fn test_duration_extension_from_float() {
        let d = <Duration as DurationExtension>::from_days_float(1.0);
        assert_eq!(d.as_secs(), 86_400);

        let d = <Duration as DurationExtension>::from_hours_float(2.5);
        assert_eq!(d.as_secs(), 9_000);

        let d = <Duration as DurationExtension>::from_minutes_float(1.5);
        assert_eq!(d.as_secs(), 90);

        let d = <Duration as DurationExtension>::from_secs_float(2.5);
        assert_eq!(d.as_secs_f64(), 2.5);

        let d = <Duration as DurationExtension>::from_millis_float(1500.0);
        assert_eq!(d.as_secs_f64(), 1.5);

        let d = <Duration as DurationExtension>::from_nanos_float(2e9);
        assert_eq!(d.as_secs(), 2);
    }

    #[test]
    fn test_duration_extension_as_float() {
        let d = Duration::from_secs(90);

        assert!(approx_eq(d.as_secs_float(), 90.0, 1e-9));
        assert!(approx_eq(d.as_minutes_float(), 1.5, 1e-9));
        assert!(approx_eq(d.as_hours_float(), 0.025, 1e-9));
        assert!(approx_eq(d.as_days_float(), 90.0 / 86400.0, 1e-9));

        let nanos = Duration::from_secs(1);
        assert!(approx_eq(nanos.as_nanos_float(), 1e9, 1e-3));

        let millis = Duration::from_secs(1);
        assert!(approx_eq(millis.as_millis_float(), 1000.0, 1e-6));
    }

    #[test]
    fn test_duration_extension_from_int() {
        let d = <Duration as DurationExtension>::from_minutes(2);
        assert_eq!(d.as_secs(), 120);

        let d = <Duration as DurationExtension>::from_hours(3);
        assert_eq!(d.as_secs(), 10_800);
    }
}
