//! Core simulation engine for discrete event simulation.

use super::acd::Acd;
use super::agent::Agent;
use super::call::Call;
use super::event::{Event, EventQueue, EventType};
use super::queue::Queue;
use super::{AgentId, CallId, CallType};
use crate::Float;
use crate::distributions::Distribution;
use crate::simulation::AgentType;
use crate::time::TimeUnit;
use rand::RngCore;
use std::fs::File;
use std::path::Path;
use std::time::Duration;

/// Result of a simulation run.
pub struct SimulationResult {
    /// All calls that were processed during the simulation.
    calls: Vec<Call>,
    /// All agents in the simulation.
    agents: Vec<Agent>,
}

impl SimulationResult {
    /// Returns a reference to all calls in the simulation.
    pub fn calls(&self) -> &[Call] {
        &self.calls
    }

    /// Returns a reference to all agents in the simulation.
    pub fn agents(&self) -> &[Agent] {
        &self.agents
    }

    /// Writes the simulation calls to a CSV file.
    ///
    /// # Arguments
    /// * `file_path` - Path to the output CSV file
    /// * `separator` - Optional separator character (defaults to comma if None)
    ///
    /// # Returns
    /// * `Ok(())` if successful
    /// * `Err` if the file already exists or if writing fails
    ///
    /// # Errors
    /// Returns an error if:
    /// - The file already exists
    /// - The file cannot be created
    /// - Writing to the file fails
    pub fn write_to_csv_file(
        &self,
        file_path: &str,
        separator: Option<char>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(serde::Serialize)]
        struct CallCsvRow {
            call_id: CallId,
            call_type: CallType,
            arrival_time_seconds: Float,
            service_start_time_seconds: String,
            service_end_time_seconds: String,
            assigned_agent_id: String,
            assigned_agent_type: String,
        }

        // Check if file already exists
        if Path::new(file_path).exists() {
            return Err(format!("File already exists: {}", file_path).into());
        }

        // Create the file
        let file = File::create(file_path)?;

        // Create CSV writer with specified separator
        let separator = separator.unwrap_or(',');
        let mut wtr = csv::WriterBuilder::new()
            .delimiter(separator as u8)
            .from_writer(file);

        // Write each call using serde serialization
        let time_unit = TimeUnit::Seconds;
        for call in &self.calls {
            let row = CallCsvRow {
                call_id: call.id(),
                call_type: call.call_type(),
                arrival_time_seconds: time_unit.from(call.arrival_time()),
                service_start_time_seconds: call
                    .service_start_time()
                    .map(|t| time_unit.from(t).to_string())
                    .unwrap_or_default(),
                service_end_time_seconds: call
                    .service_end_time()
                    .map(|t| time_unit.from(t).to_string())
                    .unwrap_or_default(),
                assigned_agent_id: call
                    .assigned_agent_id()
                    .map(|a| a.to_string())
                    .unwrap_or_default(),
                assigned_agent_type: call
                    .assigned_agent_id()
                    .and_then(|agent_id| self.agents.get(agent_id))
                    .map(|agent| agent.agent_type().to_string())
                    .unwrap_or_default(),
            };
            wtr.serialize(row)?;
        }

        wtr.flush()?;
        Ok(())
    }

    /// Writes the simulation calls to a JSON file.
    ///
    /// # Arguments
    /// * `file_path` - Path to the output JSON file
    ///
    /// # Returns
    /// * `Ok(())` if successful
    /// * `Err` if the file already exists or if writing fails
    ///
    /// # Errors
    /// Returns an error if:
    /// - The file already exists
    /// - The file cannot be created
    /// - Writing to the file fails
    pub fn write_to_json_file(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(serde::Serialize)]
        struct CallJsonRow {
            call_id: CallId,
            call_type: CallType,
            arrival_time_seconds: Float,
            service_start_time_seconds: Option<Float>,
            service_end_time_seconds: Option<Float>,
            assigned_agent_id: Option<AgentId>,
            assigned_agent_type: Option<AgentType>,
        }

        // Check if file already exists
        if Path::new(file_path).exists() {
            return Err(format!("File already exists: {}", file_path).into());
        }

        // Convert calls to JSON rows
        let time_unit = TimeUnit::Seconds;
        let rows: Vec<CallJsonRow> = self
            .calls
            .iter()
            .map(|call| CallJsonRow {
                call_id: call.id(),
                call_type: call.call_type(),
                arrival_time_seconds: time_unit.from(call.arrival_time()),
                service_start_time_seconds: call.service_start_time().map(|t| time_unit.from(t)),
                service_end_time_seconds: call.service_end_time().map(|t| time_unit.from(t)),
                assigned_agent_id: call.assigned_agent_id(),
                assigned_agent_type: call
                    .assigned_agent_id()
                    .and_then(|agent_id| self.agents.get(agent_id))
                    .map(|agent| agent.agent_type()),
            })
            .collect();

        // Serialize to JSON and write to file
        let json = serde_json::to_string(&rows)?;
        std::fs::write(file_path, json)?;

        Ok(())
    }
}

/// Runs a discrete event simulation until the specified stop time.
///
/// Uses index-based lookups for O(1) performance during simulation. Call types and
/// agent types are identified by their indices in the provided slices.
///
/// # Arguments
/// * `arrival_distributions` - Arrival time distributions for each call type.
///   Index represents the call type ID.
/// * `agents_per_type` - Number of agents to create for each agent type.
///   Index represents the agent type ID.
/// * `service_distributions` - 2D matrix of service time distributions indexed by
///   `[agent_type_idx][call_type_idx]`. Use `None` if an agent type cannot handle
///   a call type.
/// * `acd` - Automatic Call Distributor for routing decisions.
/// * `rng` - Random number generator for sampling distributions.
/// * `stop_time` - When to stop generating new call arrivals.
///
/// # Returns
/// * `SimulationResult` containing all processed calls and agents.
///
/// # Panics
/// Panics if:
/// - `arrival_distributions` is empty
/// - `agents_per_type` sums to zero (no agents)
/// - `service_distributions` dimensions don't match the number of agent types and call types
/// - Any call type cannot be handled by at least one agent type with available agents
pub fn simulate(
    arrival_distributions: &[&dyn Distribution],
    agents_per_type: &[u32],
    service_distributions: &[&[Option<&dyn Distribution>]],
    acd: &dyn Acd,
    rng: &mut dyn RngCore,
    stop_time: Duration,
) -> SimulationResult {
    // Validate arrival distributions
    assert!(
        !arrival_distributions.is_empty(),
        "At least one call type with arrival distribution is required",
    );

    // Validate agents per type
    assert!(
        agents_per_type.iter().sum::<u32>() > 0,
        "At least one agent is required in the simulation",
    );

    // Validate service distributions dimensions
    assert_eq!(
        service_distributions.len(),
        agents_per_type.len(),
        "Service distributions rows ({}) must match number of agent types ({})",
        service_distributions.len(),
        agents_per_type.len()
    );
    service_distributions.iter().enumerate().for_each(|(agent_type_idx, row)| {
        assert_eq!(
            row.len(),
            arrival_distributions.len(),
            "Service distributions columns for agent type {} ({}) must match number of call types ({})",
            agent_type_idx,
            row.len(),
            arrival_distributions.len(),
        );
    });

    // Validate that every call type can be handled by at least one agent type with agents
    (0..arrival_distributions.len()).for_each(|call_type_idx| {
        let can_be_handled =
            service_distributions
                .iter()
                .enumerate()
                .any(|(agent_type_idx, row)| {
                    row[call_type_idx].is_some() && agents_per_type[agent_type_idx] > 0
                });
        assert!(
            can_be_handled,
            "Call type {} cannot be handled: no agent type with agents can serve it",
            call_type_idx
        );
    });

    // Create agents vector
    let mut agents: Vec<Agent> = agents_per_type
        .iter()
        .enumerate()
        .flat_map(|(agent_type_idx, &count)| std::iter::repeat_n(agent_type_idx, count as usize))
        .enumerate()
        .map(|(agent_id, agent_type_idx)| Agent::new(agent_id, agent_type_idx))
        .collect();

    let mut event_queue = EventQueue::new();
    let mut call_queue = Queue::new();
    let mut calls: Vec<Call> = Vec::new();
    let mut current_time = Duration::ZERO;

    arrival_distributions
        .iter()
        .enumerate()
        .map(|(call_type, distribution)| {
            (
                call_type,
                distribution.sample(current_time, rng) + current_time,
            )
        })
        .filter(|(_, arrival_time)| *arrival_time < stop_time)
        .for_each(|(call_type, arrival_time)| {
            event_queue.push(Event::call_arrival(arrival_time, call_type))
        });

    while let Some(event) = event_queue.pop() {
        current_time = event.time();

        match event.event_type() {
            EventType::CallArrival { call_type } => {
                let call_id = calls.len();
                let mut call = Call::new(call_id, *call_type, current_time);

                let next_arrival_time =
                    arrival_distributions[*call_type].sample(current_time, rng) + current_time;
                if next_arrival_time < stop_time {
                    event_queue.push(Event::call_arrival(next_arrival_time, *call_type));
                }

                let idle_agents: Vec<&Agent> = agents
                    .iter()
                    .filter(|agent| {
                        agent.is_idle()
                            && service_distributions[agent.agent_type()][*call_type].is_some()
                    })
                    .collect();
                if let Some(agent_id) = acd.route(&call, &idle_agents) {
                    let agent = &mut agents[agent_id];
                    let agent_type = agent.agent_type();
                    call.start_service(current_time, agent_id);
                    agent.start_service(call_id);
                    let service_time = service_distributions[agent_type][*call_type]
                        .unwrap()
                        .sample(current_time, rng);
                    event_queue.push(Event::service_end(
                        current_time + service_time,
                        call_id,
                        agent_id,
                    ));
                } else {
                    call_queue.enqueue(call_id);
                }
                calls.push(call);
            }
            EventType::ServiceEnd { call_id, agent_id } => {
                calls[*call_id].end_service(current_time);
                let agent = &mut agents[*agent_id];
                let agent_type = agent.agent_type();
                agent.end_service(current_time);

                if current_time >= stop_time {
                    continue;
                }
                let Some(call_id) = call_queue.dequeue() else {
                    continue;
                };
                let call = &mut calls[call_id];
                let call_type = call.call_type();

                call.start_service(current_time, *agent_id);
                agent.start_service(call_id);
                let service_time = service_distributions[agent_type][call_type]
                    .unwrap()
                    .sample(current_time, rng);
                event_queue.push(Event::service_end(
                    current_time + service_time,
                    call_id,
                    *agent_id,
                ));
            }
        }
    }

    SimulationResult { calls, agents }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_to_csv_file() {
        // Create agents
        let agent1 = Agent::new(1, 0); // Agent ID 1, Type 0
        let agent2 = Agent::new(2, 1); // Agent ID 2, Type 1

        // Create test calls with different states
        let mut call1 = Call::new(0, 1, Duration::from_secs(10));
        call1.start_service(Duration::from_secs(15), 1);
        call1.end_service(Duration::from_secs(30));

        let mut call2 = Call::new(1, 2, Duration::from_secs(20));
        call2.start_service(Duration::from_secs(25), 2);

        let call3 = Call::new(2, 1, Duration::from_secs(35));

        let result = SimulationResult {
            calls: vec![call1, call2, call3],
            agents: vec![Agent::new(0, 0), agent1, agent2],
        };

        // Write to CSV with default separator in temp directory
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_output.csv");
        result
            .write_to_csv_file(file_path.to_str().unwrap(), None)
            .unwrap();

        // Verify file was created
        assert!(file_path.exists());

        // Read and verify contents
        let contents = std::fs::read_to_string(&file_path).unwrap();
        assert!(contents.contains("call_id,call_type,arrival_time_seconds"));
        assert!(contents.contains("assigned_agent_id"));
        assert!(contents.contains("assigned_agent_type"));
        assert!(contents.contains("0,1,10"));
        assert!(contents.contains("1,2,20"));
        assert!(contents.contains("2,1,35"));
    }

    #[test]
    fn test_write_to_csv_file_custom_separator() {
        let call = Call::new(0, 1, Duration::from_secs(5));
        let result = SimulationResult {
            calls: vec![call],
            agents: vec![],
        };

        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_output_semicolon.csv");
        result
            .write_to_csv_file(file_path.to_str().unwrap(), Some(';'))
            .unwrap();

        let contents = std::fs::read_to_string(&file_path).unwrap();
        assert!(contents.contains("call_id;call_type;arrival_time_seconds"));
    }

    #[test]
    fn test_write_to_csv_file_already_exists() {
        let result = SimulationResult {
            calls: vec![],
            agents: vec![],
        };

        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_output_exists.csv");
        // Create the file first
        std::fs::File::create(&file_path).unwrap();

        // Should return error
        let err = result.write_to_csv_file(file_path.to_str().unwrap(), None);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_write_to_json_file() {
        // Create agents
        let agent1 = Agent::new(1, 0); // Agent ID 1, Type 0
        let agent2 = Agent::new(2, 1); // Agent ID 2, Type 1

        // Create test calls with different states
        let mut call1 = Call::new(0, 1, Duration::from_secs(10));
        call1.start_service(Duration::from_secs(15), 1);
        call1.end_service(Duration::from_secs(30));

        let mut call2 = Call::new(1, 2, Duration::from_secs(20));
        call2.start_service(Duration::from_secs(25), 2);

        let call3 = Call::new(2, 1, Duration::from_secs(35));

        let result = SimulationResult {
            calls: vec![call1, call2, call3],
            agents: vec![Agent::new(0, 0), agent1, agent2],
        };

        // Write to JSON in temp directory
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_output.json");
        result
            .write_to_json_file(file_path.to_str().unwrap())
            .unwrap();

        // Verify file was created
        assert!(file_path.exists());

        // Read and verify contents
        let contents = std::fs::read_to_string(&file_path).unwrap();

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert!(parsed.is_array());

        // Verify array length
        let array = parsed.as_array().unwrap();
        assert_eq!(array.len(), 3);

        // Verify first call (completed)
        assert_eq!(array[0]["call_id"], 0);
        assert_eq!(array[0]["call_type"], 1);
        assert_eq!(array[0]["arrival_time_seconds"], 10.0);
        assert_eq!(array[0]["service_start_time_seconds"], 15.0);
        assert_eq!(array[0]["service_end_time_seconds"], 30.0);
        assert_eq!(array[0]["assigned_agent_id"], 1);
        assert_eq!(array[0]["assigned_agent_type"], 0);

        // Verify second call (in service, no end time)
        assert_eq!(array[1]["call_id"], 1);
        assert_eq!(array[1]["service_start_time_seconds"], 25.0);
        assert!(array[1]["service_end_time_seconds"].is_null());
        assert_eq!(array[1]["assigned_agent_id"], 2);
        assert_eq!(array[1]["assigned_agent_type"], 1);

        // Verify third call (waiting, no service times)
        assert_eq!(array[2]["call_id"], 2);
        assert!(array[2]["service_start_time_seconds"].is_null());
        assert!(array[2]["service_end_time_seconds"].is_null());
        assert!(array[2]["assigned_agent_id"].is_null());
        assert!(array[2]["assigned_agent_type"].is_null());
    }

    #[test]
    fn test_write_to_json_file_already_exists() {
        let result = SimulationResult {
            calls: vec![],
            agents: vec![],
        };

        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_output_exists.json");
        // Create the file first
        std::fs::File::create(&file_path).unwrap();

        // Should return error
        let err = result.write_to_json_file(file_path.to_str().unwrap());
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("already exists"));
    }

    // Helper mock distribution for validation tests
    struct MockDistribution;
    impl Distribution for MockDistribution {
        fn sample(&self, _time: Duration, _rng: &mut dyn RngCore) -> Duration {
            Duration::from_secs(1)
        }
    }

    // Helper mock ACD for validation tests
    struct MockAcd;
    impl Acd for MockAcd {
        fn route(&self, _call: &Call, idle_agents: &[&Agent]) -> Option<AgentId> {
            idle_agents.first().map(|a| a.id())
        }
    }

    #[test]
    #[should_panic(expected = "At least one call type with arrival distribution is required")]
    fn test_simulate_empty_arrival_distributions() {
        let arrival_distributions: &[&dyn Distribution] = &[];
        let agents_per_type: &[u32] = &[1];
        let service_distributions: &[&[Option<&dyn Distribution>]] = &[];
        let acd = MockAcd;
        let mut rng = rand::rng();

        simulate(
            arrival_distributions,
            agents_per_type,
            service_distributions,
            &acd,
            &mut rng,
            Duration::from_secs(100),
        );
    }

    #[test]
    #[should_panic(expected = "At least one agent is required in the simulation")]
    fn test_simulate_no_agents() {
        let dist = MockDistribution;
        let arrival_distributions: &[&dyn Distribution] = &[&dist];
        let agents_per_type: &[u32] = &[0];
        let row: &[Option<&dyn Distribution>] = &[Some(&dist)];
        let service_distributions: &[&[Option<&dyn Distribution>]] = &[row];
        let acd = MockAcd;
        let mut rng = rand::rng();

        simulate(
            arrival_distributions,
            agents_per_type,
            service_distributions,
            &acd,
            &mut rng,
            Duration::from_secs(100),
        );
    }

    #[test]
    #[should_panic(
        expected = "Service distributions rows (1) must match number of agent types (2)"
    )]
    fn test_simulate_service_distributions_wrong_rows() {
        let dist = MockDistribution;
        let arrival_distributions: &[&dyn Distribution] = &[&dist];
        let agents_per_type: &[u32] = &[1, 1];
        let row: &[Option<&dyn Distribution>] = &[Some(&dist)];
        let service_distributions: &[&[Option<&dyn Distribution>]] = &[row];
        let acd = MockAcd;
        let mut rng = rand::rng();

        simulate(
            arrival_distributions,
            agents_per_type,
            service_distributions,
            &acd,
            &mut rng,
            Duration::from_secs(100),
        );
    }

    #[test]
    #[should_panic(
        expected = "Service distributions columns for agent type 0 (1) must match number of call types (2)"
    )]
    fn test_simulate_service_distributions_wrong_columns() {
        let dist = MockDistribution;
        let arrival_distributions: &[&dyn Distribution] = &[&dist, &dist];
        let agents_per_type: &[u32] = &[1];
        let row: &[Option<&dyn Distribution>] = &[Some(&dist)];
        let service_distributions: &[&[Option<&dyn Distribution>]] = &[row];
        let acd = MockAcd;
        let mut rng = rand::rng();

        simulate(
            arrival_distributions,
            agents_per_type,
            service_distributions,
            &acd,
            &mut rng,
            Duration::from_secs(100),
        );
    }

    #[test]
    #[should_panic(
        expected = "Call type 0 cannot be handled: no agent type with agents can serve it"
    )]
    fn test_simulate_call_type_cannot_be_handled_no_distribution() {
        let dist = MockDistribution;
        let arrival_distributions: &[&dyn Distribution] = &[&dist];
        let agents_per_type: &[u32] = &[1];
        let row: &[Option<&dyn Distribution>] = &[None];
        let service_distributions: &[&[Option<&dyn Distribution>]] = &[row];
        let acd = MockAcd;
        let mut rng = rand::rng();

        simulate(
            arrival_distributions,
            agents_per_type,
            service_distributions,
            &acd,
            &mut rng,
            Duration::from_secs(100),
        );
    }

    #[test]
    #[should_panic(
        expected = "Call type 0 cannot be handled: no agent type with agents can serve it"
    )]
    fn test_simulate_call_type_cannot_be_handled_no_agents_of_type() {
        let dist = MockDistribution;
        let arrival_distributions: &[&dyn Distribution] = &[&dist];
        let agents_per_type: &[u32] = &[0, 1];
        let row1: &[Option<&dyn Distribution>] = &[Some(&dist)];
        let row2: &[Option<&dyn Distribution>] = &[None];
        let service_distributions: &[&[Option<&dyn Distribution>]] = &[row1, row2];
        let acd = MockAcd;
        let mut rng = rand::rng();

        simulate(
            arrival_distributions,
            agents_per_type,
            service_distributions,
            &acd,
            &mut rng,
            Duration::from_secs(100),
        );
    }
}
