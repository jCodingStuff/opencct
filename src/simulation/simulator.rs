//! Core simulation engine for discrete event simulation.

use super::acd::Acd;
use super::agent::Agent;
use super::call::Call;
use super::{AgentType, CallType};
use crate::distributions::Distribution;
use std::collections::HashMap;
use std::time::Duration;

/// Configuration for a simulation run.
pub struct SimulationConfig {
    /// Initial agents in the system.
    pub agents: Vec<Agent>,

    /// Arrival time distributions for each call type.
    /// Generates inter-arrival times (time between consecutive calls).
    pub arrival_distributions: HashMap<CallType, Box<dyn Distribution>>,

    /// Service time distributions for each (agent_type, call_type) pair.
    /// Generates how long it takes an agent type to serve a call type.
    pub service_distributions: HashMap<(AgentType, CallType), Box<dyn Distribution>>,

    /// Automatic Call Distributor for routing decisions.
    pub acd: Box<dyn Acd>,
}

/// Result of a simulation run.
pub struct SimulationResult {
    /// All calls that were processed during the simulation.
    calls: Vec<Call>,
}

impl SimulationResult {
    /// Returns a reference to all calls in the simulation.
    pub fn calls(&self) -> &[Call] {
        &self.calls
    }
}

/// Runs a discrete event simulation until the specified stop time.
///
/// # Arguments
/// * `config` - Simulation configuration (agents, distributions, ACD)
/// * `stop_time` - When to stop the simulation
///
/// # Returns
/// * `SimulationResult` containing all processed calls and statistics
pub fn simulate(_config: SimulationConfig, _stop_time: Duration) -> SimulationResult {
    // TODO: Implement simulation logic
    SimulationResult { calls: Vec::new() }
}
