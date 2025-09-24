//! Utilities and extensions for time.

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
        #[cfg(feature = "f32")]
        { Duration::from_secs_32(value * self.factor()) }

        #[cfg(feature = "f64")]
        { Duration::from_secs_f64(value * self.factor()) }
    }

    /// Convert a [Duration] to a [Float] value
    /// # Arguments
    /// * `d` - The [Duration]
    /// # Returns
    /// The value interpreted by the unit as [Float]
    pub fn from_duration(&self, d: Duration) -> Float {
        #[cfg(feature = "f32")]
        { d.as_secs_f32() / self.factor() }

        #[cfg(feature = "f64")]
        { d.as_secs_f64() / self.factor() }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::Float;

    #[test]
    fn test_factors() {
        assert_eq!(TimeUnit::Days.factor(), 86400.0);
        assert_eq!(TimeUnit::Hours.factor(), 3600.0);
        assert_eq!(TimeUnit::Minutes.factor(), 60.0);
        assert_eq!(TimeUnit::Seconds.factor(), 1.0);
        assert_eq!(TimeUnit::Millis.factor(), 1e-3);
        assert_eq!(TimeUnit::Nanos.factor(), 1e-9);
    }

    #[test]
    fn test_to_duration_seconds() {
        let d = TimeUnit::Seconds.to_duration(5.0 as Float);
        let total: Float = d.as_secs() as Float + (d.subsec_nanos() as Float) * 1e-9;
        assert!((total - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_to_duration_minutes() {
        let d = TimeUnit::Minutes.to_duration(2.0 as Float);
        let total: Float = d.as_secs() as Float + (d.subsec_nanos() as Float) * 1e-9;
        assert!((total - 120.0).abs() < 1e-9);
    }

    #[test]
    fn test_from_duration_roundtrip() {
        let unit = TimeUnit::Hours;
        let value: Float = 3.5;
        let dur = unit.to_duration(value);
        let back: Float = unit.from_duration(dur);
        assert!((back - value).abs() < 1e-6, "expected {value}, got {back}");
    }

    #[test]
    fn test_zero_value() {
        let unit = TimeUnit::Millis;
        let d = unit.to_duration(0.0 as Float);
        let back: Float = unit.from_duration(d);
        assert!((back - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_large_value() {
        let unit = TimeUnit::Days;
        let value: Float = 1000.0; // ~1000 days
        let d = unit.to_duration(value);
        let back: Float = unit.from_duration(d);
        assert!((back - value).abs() < 1e-6, "large roundtrip failed: {value} vs {back}");
    }

    #[test]
    fn test_fractional_millis() {
        let unit = TimeUnit::Millis;
        let value: Float = 1.5; // 1.5 ms
        let d = unit.to_duration(value);
        let back: Float = unit.from_duration(d);
        assert!((back - value).abs() < 1e-9, "fractional millis roundtrip failed: {value} vs {back}");
    }

    #[test]
    fn test_fractional_nanos() {
        let unit = TimeUnit::Nanos;
        let value: Float = 2500.0; // 2500 ns = 2.5 μs
        let d = unit.to_duration(value);
        let back: Float = unit.from_duration(d);
        assert!((back - value).abs() < 1e-9, "fractional nanos roundtrip failed: {value} vs {back}");
    }
}
