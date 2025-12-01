//! Event types and event queue for discrete event simulation.

use super::{AgentId, CallId, CallType};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Duration;

/// Types of events that can occur in the simulation.
#[derive(Debug, Clone)]
pub enum EventType {
    /// A new call arrives at the system and enters the queue.
    CallArrival { call_type: CallType },
    /// An agent finishes serving a call (agent becomes idle).
    ServiceEnd { call_id: CallId, agent_id: AgentId },
}

/// Represents a discrete event in the simulation.
/// Events are scheduled to occur at specific times.
#[derive(Debug, Clone)]
pub struct Event {
    /// Time when this event occurs.
    time: Duration,
    /// Type of event and its associated data.
    event_type: EventType,
}

impl Event {
    /// Creates a new call arrival event.
    pub fn call_arrival(time: Duration, call_type: CallType) -> Self {
        Self {
            time,
            event_type: EventType::CallArrival { call_type },
        }
    }

    /// Creates a new service end event.
    pub fn service_end(time: Duration, call_id: CallId, agent_id: AgentId) -> Self {
        Self {
            time,
            event_type: EventType::ServiceEnd { call_id, agent_id },
        }
    }

    /// Returns the time when this event occurs.
    pub fn time(&self) -> Duration {
        self.time
    }

    /// Returns a reference to the event type and its associated data.
    pub fn event_type(&self) -> &EventType {
        &self.event_type
    }
}

// Implement ordering for events based on time.
// Since BinaryHeap is a max-heap, we reverse the ordering to get a min-heap (earliest event first).

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering: earlier times compare as "greater" for min-heap behavior
        other.time.cmp(&self.time)
    }
}

/// Priority queue for managing events in chronological order.
/// Uses a BinaryHeap internally to efficiently retrieve the next event.
pub struct EventQueue {
    heap: BinaryHeap<Event>,
}

impl EventQueue {
    /// Creates a new empty event queue.
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    /// Adds an event to the queue.
    pub fn push(&mut self, event: Event) {
        self.heap.push(event);
    }

    /// Removes and returns the next event (earliest time).
    /// Returns `None` if the queue is empty.
    pub fn pop(&mut self) -> Option<Event> {
        self.heap.pop()
    }

    /// Returns the number of events in the queue.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

impl Default for EventQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_ordering() {
        // Create events at different times
        let e1 = Event::call_arrival(Duration::from_secs(10), 0);
        let e2 = Event::call_arrival(Duration::from_secs(5), 0);
        let e3 = Event::call_arrival(Duration::from_secs(15), 0);

        let mut queue = EventQueue::new();
        queue.push(e1);
        queue.push(e2);
        queue.push(e3);

        // Should pop events in chronological order (earliest first)
        assert_eq!(queue.pop().unwrap().time(), Duration::from_secs(5));
        assert_eq!(queue.pop().unwrap().time(), Duration::from_secs(10));
        assert_eq!(queue.pop().unwrap().time(), Duration::from_secs(15));
        assert!(queue.pop().is_none());
    }

    #[test]
    fn test_event_queue_length() {
        let mut queue = EventQueue::new();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());

        queue.push(Event::call_arrival(Duration::from_secs(1), 0));
        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());

        queue.push(Event::service_end(Duration::from_secs(2), 0, 0));
        assert_eq!(queue.len(), 2);

        queue.pop();
        assert_eq!(queue.len(), 1);

        queue.pop();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_event_types() {
        let call_arrival = Event::call_arrival(Duration::from_secs(1), 3);
        match call_arrival.event_type() {
            EventType::CallArrival { call_type } => {
                assert_eq!(*call_type, 3);
            }
            _ => panic!("Expected CallArrival event"),
        }

        let service_end = Event::service_end(Duration::from_secs(2), 10, 5);
        match service_end.event_type() {
            EventType::ServiceEnd { call_id, agent_id } => {
                assert_eq!(*call_id, 10);
                assert_eq!(*agent_id, 5);
            }
            _ => panic!("Expected ServiceEnd event"),
        }
    }
}
