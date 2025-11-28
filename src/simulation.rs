//! Discrete event simulation engine for call center modeling.

/// Unique identifier for a call in the simulation.
/// Each call that enters the system gets a unique ID (0, 1, 2, ...).
pub type CallId = usize;

/// Unique identifier for an agent in the simulation.
/// Each agent gets a unique ID (0, 1, 2, ...).
pub type AgentId = usize;

/// Type identifier for different call types.
/// For example: 0 = Sales, 1 = Support, 2 = Billing
pub type CallType = usize;

/// Type identifier for different agent types/skill groups.
/// For example: 0 = Junior, 1 = Senior, 2 = Specialist
pub type AgentType = usize;

pub mod acd;
pub mod agent;
pub mod call;
pub mod event;
pub mod queue;
pub mod simulator;
