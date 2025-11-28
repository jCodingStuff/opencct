//! Agent entity for discrete event simulation.

use super::{AgentId, AgentType, CallId};
use std::time::Duration;

/// Represents an agent who can serve calls.
#[derive(Debug, Clone)]
pub struct Agent {
    id: AgentId,
    agent_type: AgentType,
    /// The call currently being served, if any.
    current_call: Option<CallId>,
    /// Time when the agent became idle (None if currently busy).
    idle_since: Option<Duration>,
}

impl Agent {
    /// Creates a new agent, idle since the given time.
    pub fn new(id: AgentId, agent_type: AgentType, idle_since: Duration) -> Self {
        Self {
            id,
            agent_type,
            current_call: None,
            idle_since: Some(idle_since),
        }
    }

    // Getters for read-only access

    /// Returns the unique ID of this agent.
    pub fn id(&self) -> AgentId {
        self.id
    }

    /// Returns the type/skill group of this agent.
    pub fn agent_type(&self) -> AgentType {
        self.agent_type
    }

    /// Returns `true` if the agent is idle (not serving a call).
    pub fn is_idle(&self) -> bool {
        self.current_call.is_none()
    }

    /// Returns `true` if the agent is busy (serving a call).
    pub fn is_busy(&self) -> bool {
        self.current_call.is_some()
    }

    /// Returns the call currently being served, if any.
    pub fn current_call(&self) -> Option<CallId> {
        self.current_call
    }

    /// Returns the time when this agent became idle, if currently idle.
    pub fn idle_since(&self) -> Option<Duration> {
        self.idle_since
    }

    // State transition methods

    /// Starts serving a call.
    /// # Panics
    /// Panics if the agent is already busy.
    pub fn start_service(&mut self, call_id: CallId) {
        assert!(self.is_idle(), "Agent is already busy serving a call");
        self.current_call = Some(call_id);
        self.idle_since = None;
    }

    /// Ends service for the current call, making the agent idle.
    /// # Panics
    /// Panics if the agent is not currently serving a call.
    pub fn end_service(&mut self, time: Duration) {
        assert!(
            self.is_busy(),
            "Cannot end service - agent is not serving a call"
        );
        self.current_call = None;
        self.idle_since = Some(time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_agent() {
        let agent = Agent::new(5, 0, Duration::from_secs(0));
        assert_eq!(agent.id(), 5);
        assert_eq!(agent.agent_type(), 0);
        assert!(agent.is_idle());
        assert!(!agent.is_busy());
        assert_eq!(agent.current_call(), None);
        assert_eq!(agent.idle_since(), Some(Duration::from_secs(0)));
    }

    #[test]
    fn test_agent_state_transitions() {
        let mut agent = Agent::new(0, 0, Duration::from_secs(0));

        // Agent starts idle
        assert!(agent.is_idle());
        assert_eq!(agent.idle_since(), Some(Duration::from_secs(0)));

        // Start serving call 42
        agent.start_service(42);
        assert!(agent.is_busy());
        assert!(!agent.is_idle());
        assert_eq!(agent.current_call(), Some(42));
        assert_eq!(agent.idle_since(), None);

        // End service at t=100
        agent.end_service(Duration::from_secs(100));
        assert!(agent.is_idle());
        assert!(!agent.is_busy());
        assert_eq!(agent.current_call(), None);
        assert_eq!(agent.idle_since(), Some(Duration::from_secs(100)));
    }

    #[test]
    fn test_agent_idle_since_tracking() {
        let mut agent = Agent::new(0, 0, Duration::from_secs(10));
        assert_eq!(agent.idle_since(), Some(Duration::from_secs(10)));

        agent.start_service(1);
        assert_eq!(agent.idle_since(), None);

        agent.end_service(Duration::from_secs(50));
        assert_eq!(agent.idle_since(), Some(Duration::from_secs(50)));

        agent.start_service(2);
        agent.end_service(Duration::from_secs(200));
        assert_eq!(agent.idle_since(), Some(Duration::from_secs(200)));
    }

    #[test]
    #[should_panic(expected = "Agent is already busy serving a call")]
    fn test_agent_cannot_start_service_when_busy() {
        let mut agent = Agent::new(0, 0, Duration::from_secs(0));
        agent.start_service(1);
        agent.start_service(2); // Should panic
    }

    #[test]
    #[should_panic(expected = "Cannot end service - agent is not serving a call")]
    fn test_agent_cannot_end_service_when_idle() {
        let mut agent = Agent::new(0, 0, Duration::from_secs(0));
        agent.end_service(Duration::from_secs(10)); // Should panic
    }
}
