//! Call entity for discrete event simulation.

use super::event::{AgentId, CallId, CallType};
use std::time::Duration;

/// Represents a call in the system.
#[derive(Debug, Clone)]
pub struct Call {
    id: CallId,
    call_type: CallType,
    arrival_time: Duration,
    service_start_time: Option<Duration>,
    service_end_time: Option<Duration>,
    assigned_agent: Option<AgentId>,
}

impl Call {
    /// Creates a new call that just arrived.
    pub fn new(id: CallId, call_type: CallType, arrival_time: Duration) -> Self {
        Call {
            id,
            call_type,
            arrival_time,
            service_start_time: None,
            service_end_time: None,
            assigned_agent: None,
        }
    }

    // Getters for read-only access

    /// Returns the unique ID of this call.
    pub fn id(&self) -> CallId {
        self.id
    }

    /// Returns the type of this call.
    pub fn call_type(&self) -> CallType {
        self.call_type
    }

    /// Returns the time when this call arrived.
    pub fn arrival_time(&self) -> Duration {
        self.arrival_time
    }

    /// Returns the time when service started, if it has started.
    pub fn service_start_time(&self) -> Option<Duration> {
        self.service_start_time
    }

    /// Returns the time when service ended, if it has ended.
    pub fn service_end_time(&self) -> Option<Duration> {
        self.service_end_time
    }

    /// Returns the agent assigned to this call, if any.
    pub fn assigned_agent(&self) -> Option<AgentId> {
        self.assigned_agent
    }

    // State transition methods

    /// Starts service for this call.
    /// # Panics
    /// Panics if service has already started.
    pub fn start_service(&mut self, time: Duration, agent_id: AgentId) {
        assert!(
            self.service_start_time.is_none(),
            "Cannot start service that has already started"
        );
        self.service_start_time = Some(time);
        self.assigned_agent = Some(agent_id);
    }

    /// Ends service for this call.
    /// # Panics
    /// Panics if service hasn't started or has already ended.
    pub fn end_service(&mut self, time: Duration) {
        assert!(
            self.service_start_time.is_some(),
            "Cannot end service that hasn't started"
        );
        assert!(self.service_end_time.is_none(), "Service has already ended");
        self.service_end_time = Some(time);
    }

    // Metrics

    /// Calculates wait time (time spent in queue before service started).
    /// Returns None if service hasn't started yet.
    pub fn wait_time(&self) -> Option<Duration> {
        self.service_start_time
            .map(|start| start.saturating_sub(self.arrival_time))
    }

    /// Calculates service duration (time spent being served).
    /// Returns None if service hasn't finished yet.
    pub fn service_duration(&self) -> Option<Duration> {
        match (self.service_start_time, self.service_end_time) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        }
    }

    /// Calculates total time in system (from arrival to completion).
    /// Returns None if call hasn't finished yet.
    pub fn total_time(&self) -> Option<Duration> {
        self.service_end_time
            .map(|end| end.saturating_sub(self.arrival_time))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_call() {
        let call = Call::new(42, 1, Duration::from_secs(10));
        assert_eq!(call.id(), 42);
        assert_eq!(call.call_type(), 1);
        assert_eq!(call.arrival_time(), Duration::from_secs(10));
        assert!(call.service_start_time().is_none());
        assert!(call.service_end_time().is_none());
        assert!(call.assigned_agent().is_none());
    }

    #[test]
    fn test_state_transitions() {
        let mut call = Call::new(0, 0, Duration::from_secs(5));

        // Start service at t=10 with agent 3
        call.start_service(Duration::from_secs(10), 3);
        assert_eq!(call.service_start_time(), Some(Duration::from_secs(10)));
        assert_eq!(call.assigned_agent(), Some(3));
        assert_eq!(call.wait_time(), Some(Duration::from_secs(5)));

        // End service at t=25
        call.end_service(Duration::from_secs(25));
        assert_eq!(call.service_end_time(), Some(Duration::from_secs(25)));
        assert_eq!(call.service_duration(), Some(Duration::from_secs(15)));
        assert_eq!(call.total_time(), Some(Duration::from_secs(20)));
    }

    #[test]
    #[should_panic(expected = "Cannot start service that has already started")]
    fn test_cannot_start_service_twice() {
        let mut call = Call::new(0, 0, Duration::from_secs(5));
        call.start_service(Duration::from_secs(10), 0);
        call.start_service(Duration::from_secs(15), 1); // Should panic
    }

    #[test]
    #[should_panic(expected = "Cannot end service that hasn't started")]
    fn test_cannot_end_service_before_start() {
        let mut call = Call::new(0, 0, Duration::from_secs(5));
        call.end_service(Duration::from_secs(10)); // Should panic
    }

    #[test]
    #[should_panic(expected = "Service has already ended")]
    fn test_cannot_end_service_twice() {
        let mut call = Call::new(0, 0, Duration::from_secs(5));
        call.start_service(Duration::from_secs(10), 0);
        call.end_service(Duration::from_secs(25));
        call.end_service(Duration::from_secs(30)); // Should panic
    }
}
