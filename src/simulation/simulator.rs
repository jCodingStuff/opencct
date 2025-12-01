//! Core simulation engine for discrete event simulation.

use super::acd::Acd;
use super::agent::Agent;
use super::call::Call;
use super::event::{Event, EventQueue, EventType};
use super::queue::Queue;
use super::{AgentId, AgentType, CallId, CallType};
use crate::Float;
use crate::distributions::Distribution;
use crate::time::TimeUnit;
use rand::RngCore;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
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
            assigned_agent: String,
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
                assigned_agent: call
                    .assigned_agent()
                    .map(|a| a.to_string())
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
            assigned_agent: Option<AgentId>,
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
                assigned_agent: call.assigned_agent(),
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
/// # Arguments
/// * `config` - Simulation configuration (agents, distributions, ACD)
/// * `stop_time` - When to stop the simulation
/// * `rng` - Random number generator
///
/// # Returns
/// * `SimulationResult` containing all processed calls and statistics
pub fn simulate(
    config: SimulationConfig,
    stop_time: Duration,
    rng: &mut dyn RngCore,
) -> SimulationResult {
    let mut agents = config.agents;
    let mut event_queue = EventQueue::new();
    let mut call_queue = Queue::new();
    let mut calls: Vec<Call> = Vec::new();
    let mut current_time = Duration::ZERO;

    config
        .arrival_distributions
        .iter()
        .map(|(call_type, distribution)| {
            (
                *call_type,
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

                let arrival_distribution = config.arrival_distributions.get(call_type).unwrap();
                let next_arrival_time =
                    arrival_distribution.sample(current_time, rng) + current_time;
                if next_arrival_time < stop_time {
                    event_queue.push(Event::call_arrival(next_arrival_time, *call_type));
                }

                let idle_agents: Vec<&Agent> =
                    agents.iter().filter(|agent| agent.is_idle()).collect();
                if let Some(agent_id) = config.acd.route(&call, &idle_agents) {
                    call.start_service(current_time, agent_id);
                    agents[agent_id].start_service(call_id);
                    let service_distribution = config
                        .service_distributions
                        .get(&(agents[agent_id].agent_type(), call.call_type()))
                        .unwrap();
                    let service_time = service_distribution.sample(current_time, rng);
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
                agents[*agent_id].end_service(current_time);

                if current_time >= stop_time {
                    continue;
                }
                let Some(call_id) = call_queue.dequeue() else {
                    continue;
                };
                let call = &mut calls[call_id];

                call.start_service(current_time, *agent_id);
                agents[*agent_id].start_service(call_id);
                let service_distribution = config
                    .service_distributions
                    .get(&(agents[*agent_id].agent_type(), call.call_type()))
                    .unwrap();
                let service_time = service_distribution.sample(current_time, rng);
                event_queue.push(Event::service_end(
                    current_time + service_time,
                    call_id,
                    *agent_id,
                ));
            }
        }
    }

    SimulationResult { calls }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_to_csv_file() {
        // Create test calls with different states
        let mut call1 = Call::new(0, 1, Duration::from_secs(10));
        call1.start_service(Duration::from_secs(15), 1);
        call1.end_service(Duration::from_secs(30));

        let mut call2 = Call::new(1, 2, Duration::from_secs(20));
        call2.start_service(Duration::from_secs(25), 2);

        let call3 = Call::new(2, 1, Duration::from_secs(35));

        let result = SimulationResult {
            calls: vec![call1, call2, call3],
        };

        // Write to CSV with default separator
        let file_path = "test_output.csv";
        result.write_to_csv_file(file_path, None).unwrap();

        // Verify file was created
        assert!(Path::new(file_path).exists());

        // Read and verify contents
        let contents = std::fs::read_to_string(file_path).unwrap();
        assert!(contents.contains("call_id,call_type,arrival_time_seconds"));
        assert!(contents.contains("0,1,10"));
        assert!(contents.contains("1,2,20"));
        assert!(contents.contains("2,1,35"));

        // Clean up
        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_write_to_csv_file_custom_separator() {
        let call = Call::new(0, 1, Duration::from_secs(5));
        let result = SimulationResult { calls: vec![call] };

        let file_path = "test_output_semicolon.csv";
        result.write_to_csv_file(file_path, Some(';')).unwrap();

        let contents = std::fs::read_to_string(file_path).unwrap();
        assert!(contents.contains("call_id;call_type;arrival_time_seconds"));

        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_write_to_csv_file_already_exists() {
        let result = SimulationResult { calls: vec![] };

        let file_path = "test_output_exists.csv";
        // Create the file first
        std::fs::File::create(file_path).unwrap();

        // Should return error
        let err = result.write_to_csv_file(file_path, None);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("already exists"));

        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_write_to_json_file() {
        // Create test calls with different states
        let mut call1 = Call::new(0, 1, Duration::from_secs(10));
        call1.start_service(Duration::from_secs(15), 1);
        call1.end_service(Duration::from_secs(30));

        let mut call2 = Call::new(1, 2, Duration::from_secs(20));
        call2.start_service(Duration::from_secs(25), 2);

        let call3 = Call::new(2, 1, Duration::from_secs(35));

        let result = SimulationResult {
            calls: vec![call1, call2, call3],
        };

        // Write to JSON
        let file_path = "test_output.json";
        result.write_to_json_file(file_path).unwrap();

        // Verify file was created
        assert!(Path::new(file_path).exists());

        // Read and verify contents
        let contents = std::fs::read_to_string(file_path).unwrap();

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
        assert_eq!(array[0]["assigned_agent"], 1);

        // Verify second call (in service, no end time)
        assert_eq!(array[1]["call_id"], 1);
        assert_eq!(array[1]["service_start_time_seconds"], 25.0);
        assert!(array[1]["service_end_time_seconds"].is_null());

        // Verify third call (waiting, no service times)
        assert_eq!(array[2]["call_id"], 2);
        assert!(array[2]["service_start_time_seconds"].is_null());
        assert!(array[2]["service_end_time_seconds"].is_null());
        assert!(array[2]["assigned_agent"].is_null());

        // Clean up
        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_write_to_json_file_already_exists() {
        let result = SimulationResult { calls: vec![] };

        let file_path = "test_output_exists.json";
        // Create the file first
        std::fs::File::create(file_path).unwrap();

        // Should return error
        let err = result.write_to_json_file(file_path);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("already exists"));

        std::fs::remove_file(file_path).unwrap();
    }
}
