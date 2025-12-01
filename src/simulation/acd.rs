//! Automatic Call Distributor (ACD) trait and implementations.

use super::AgentId;
use super::agent::Agent;
use super::call::Call;

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
    fn route(&self, call: &Call, idle_agents: &[&Agent]) -> Option<AgentId>;
}

/// Simple FIFO + Longest Idle ACD implementation.
///
/// For new calls: assigns to the agent who has been idle the longest.
/// For idle agents: assigns the call that has been waiting the longest (FIFO).
#[derive(Clone, Debug, Default)]
pub struct FifoLongestIdleAcd;

impl FifoLongestIdleAcd {
    /// Creates a new FIFO + Longest Idle ACD.
    pub fn new() -> Self {
        Self
    }
}

impl Acd for FifoLongestIdleAcd {
    fn route(&self, _call: &Call, idle_agents: &[&Agent]) -> Option<AgentId> {
        // Find the agent who has been idle the longest
        idle_agents
            .iter()
            .filter(|agent| agent.is_idle())
            .min_by_key(|agent| agent.idle_since())
            .map(|agent| agent.id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_route_to_longest_idle() {
        let acd = FifoLongestIdleAcd::new();

        let mut agent1 = Agent::new(0, 0);
        agent1.start_service(0);
        agent1.end_service(Duration::from_secs(100));

        let mut agent2 = Agent::new(1, 0);
        agent2.start_service(1);
        agent2.end_service(Duration::from_secs(50)); // Idle longer ago

        let mut agent3 = Agent::new(2, 0);
        agent3.start_service(2);
        agent3.end_service(Duration::from_secs(200));

        let call = Call::new(0, 0, Duration::from_secs(300));

        let idle_agents = vec![&agent1, &agent2, &agent3];

        // Should pick agent2 (idle since t=50, longest time idle)
        let result = acd.route(&call, &idle_agents);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_route_no_agents() {
        let acd = FifoLongestIdleAcd::new();
        let call = Call::new(0, 0, Duration::from_secs(100));

        let result = acd.route(&call, &[]);
        assert_eq!(result, None);
    }
}
