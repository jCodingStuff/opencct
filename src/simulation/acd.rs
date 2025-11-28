//! Automatic Call Distributor (ACD) trait and implementations.

use super::agent::Agent;
use super::call::Call;
use super::{AgentId, CallId};

/// Trait for implementing Automatic Call Distributor (ACD) routing strategies.
///
/// The ACD is responsible for matching calls to agents based on custom logic.
/// Implementations can define routing strategies like longest idle first,
/// skill-based routing, priority routing, etc.
pub trait Acd {
    /// Called when a new call arrives. Try to find an idle agent for it.
    ///
    /// # Arguments
    /// * `call` - The newly arrived call
    /// * `idle_agents` - List of currently idle agents
    ///
    /// # Returns
    /// * `Some(agent_id)` if a match is found
    /// * `None` if no suitable agent is available (call goes to queue)
    fn route_new_call(&self, call: &Call, idle_agents: &[&Agent]) -> Option<AgentId>;

    /// Called when an agent becomes idle. Try to find a waiting call for them.
    ///
    /// # Arguments
    /// * `agent` - The newly idle agent
    /// * `queued_calls` - List of calls waiting in queue
    ///
    /// # Returns
    /// * `Some(call_id)` if a match is found
    /// * `None` if no suitable call is waiting (agent stays idle)
    fn route_idle_agent(&self, agent: &Agent, queued_calls: &[&Call]) -> Option<CallId>;
}

/// Simple FIFO + Longest Idle ACD implementation.
///
/// For new calls: assigns to the agent who has been idle the longest.
/// For idle agents: assigns the call that has been waiting the longest (FIFO).
#[derive(Debug, Default)]
pub struct FifoLongestIdleAcd;

impl FifoLongestIdleAcd {
    /// Creates a new FIFO + Longest Idle ACD.
    pub fn new() -> Self {
        Self
    }
}

impl Acd for FifoLongestIdleAcd {
    fn route_new_call(&self, _call: &Call, idle_agents: &[&Agent]) -> Option<AgentId> {
        // Find the agent who has been idle the longest
        idle_agents
            .iter()
            .filter(|agent| agent.is_idle())
            .min_by_key(|agent| agent.idle_since())
            .map(|agent| agent.id())
    }

    fn route_idle_agent(&self, _agent: &Agent, queued_calls: &[&Call]) -> Option<CallId> {
        // Take the first call in queue (FIFO - earliest arrival)
        queued_calls
            .iter()
            .min_by_key(|call| call.arrival_time())
            .map(|call| call.id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_route_new_call_to_longest_idle() {
        let acd = FifoLongestIdleAcd::new();

        let agent1 = Agent::new(0, 0, Duration::from_secs(100));
        let agent2 = Agent::new(1, 0, Duration::from_secs(50)); // Idle longer ago
        let agent3 = Agent::new(2, 0, Duration::from_secs(200));

        let call = Call::new(0, 0, Duration::from_secs(300));

        let idle_agents = vec![&agent1, &agent2, &agent3];

        // Should pick agent2 (idle since t=50, longest time idle)
        let result = acd.route_new_call(&call, &idle_agents);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_route_new_call_no_agents() {
        let acd = FifoLongestIdleAcd::new();
        let call = Call::new(0, 0, Duration::from_secs(100));

        let result = acd.route_new_call(&call, &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_route_idle_agent_fifo() {
        let acd = FifoLongestIdleAcd::new();

        let call1 = Call::new(0, 0, Duration::from_secs(100));
        let call2 = Call::new(1, 0, Duration::from_secs(50)); // Arrived first
        let call3 = Call::new(2, 0, Duration::from_secs(200));

        let agent = Agent::new(0, 0, Duration::from_secs(300));

        let queued_calls = vec![&call1, &call2, &call3];

        // Should pick call2 (arrived at t=50, earliest)
        let result = acd.route_idle_agent(&agent, &queued_calls);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_route_idle_agent_no_calls() {
        let acd = FifoLongestIdleAcd::new();
        let agent = Agent::new(0, 0, Duration::from_secs(100));

        let result = acd.route_idle_agent(&agent, &[]);
        assert_eq!(result, None);
    }
}
