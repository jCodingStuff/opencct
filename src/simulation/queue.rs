//! Queue entity for discrete event simulation.

use super::CallId;
use std::collections::VecDeque;

/// Represents a queue of waiting calls.
/// Uses FIFO (First In, First Out) discipline by default.
#[derive(Debug, Clone)]
pub struct Queue {
    /// The calls waiting in this queue (stored as IDs).
    calls: VecDeque<CallId>,
    /// Maximum queue length observed during simulation.
    peak_length: usize,
}

impl Queue {
    /// Creates a new empty queue.
    pub fn new() -> Self {
        Self {
            calls: VecDeque::new(),
            peak_length: 0,
        }
    }

    /// Adds a call to the back of the queue.
    pub fn enqueue(&mut self, call_id: CallId) {
        self.calls.push_back(call_id);
        self.peak_length = self.peak_length.max(self.calls.len());
    }

    /// Removes and returns the call at the front of the queue.
    /// Returns `None` if the queue is empty.
    pub fn dequeue(&mut self) -> Option<CallId> {
        self.calls.pop_front()
    }

    /// Returns the call at the front of the queue without removing it.
    /// Returns `None` if the queue is empty.
    pub fn peek(&self) -> Option<CallId> {
        self.calls.front().copied()
    }

    /// Returns the current number of calls in the queue.
    pub fn len(&self) -> usize {
        self.calls.len()
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Returns the maximum queue length observed.
    pub fn peak_length(&self) -> usize {
        self.peak_length
    }

    /// Returns an iterator over the call IDs in the queue.
    pub fn iter(&self) -> impl Iterator<Item = &CallId> {
        self.calls.iter()
    }

    /// Removes a specific call from the queue (e.g., for call abandonment).
    /// Returns `true` if the call was found and removed, `false` otherwise.
    pub fn remove(&mut self, call_id: CallId) -> bool {
        if let Some(pos) = self.calls.iter().position(|&id| id == call_id) {
            self.calls.remove(pos);
            true
        } else {
            false
        }
    }
}

impl Default for Queue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_queue() {
        let queue = Queue::new();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
        assert_eq!(queue.peak_length(), 0);
    }

    #[test]
    fn test_queue_enqueue_dequeue() {
        let mut queue = Queue::new();

        // Enqueue three calls
        queue.enqueue(10);
        queue.enqueue(20);
        queue.enqueue(30);

        assert_eq!(queue.len(), 3);
        assert!(!queue.is_empty());
        assert_eq!(queue.peak_length(), 3);

        // Dequeue should follow FIFO order
        assert_eq!(queue.dequeue(), Some(10));
        assert_eq!(queue.dequeue(), Some(20));
        assert_eq!(queue.len(), 1);

        assert_eq!(queue.dequeue(), Some(30));
        assert!(queue.is_empty());
        assert_eq!(queue.dequeue(), None);

        // Peak length should remember the maximum
        assert_eq!(queue.peak_length(), 3);
    }

    #[test]
    fn test_queue_peek() {
        let mut queue = Queue::new();

        assert_eq!(queue.peek(), None);

        queue.enqueue(100);
        queue.enqueue(200);

        // Peek should return front element without removing it
        assert_eq!(queue.peek(), Some(100));
        assert_eq!(queue.len(), 2);

        // Peek should still return same element
        assert_eq!(queue.peek(), Some(100));

        queue.dequeue();
        assert_eq!(queue.peek(), Some(200));
    }

    #[test]
    fn test_queue_peak_length() {
        let mut queue = Queue::new();

        queue.enqueue(1);
        assert_eq!(queue.peak_length(), 1);

        queue.enqueue(2);
        queue.enqueue(3);
        assert_eq!(queue.peak_length(), 3);

        queue.dequeue();
        queue.dequeue();
        assert_eq!(queue.len(), 1);
        // Peak should still remember the maximum
        assert_eq!(queue.peak_length(), 3);

        queue.enqueue(4);
        queue.enqueue(5);
        queue.enqueue(6);
        assert_eq!(queue.len(), 4);
        // New peak!
        assert_eq!(queue.peak_length(), 4);
    }

    #[test]
    fn test_queue_remove() {
        let mut queue = Queue::new();

        queue.enqueue(10);
        queue.enqueue(20);
        queue.enqueue(30);
        queue.enqueue(40);

        // Remove call from middle
        assert!(queue.remove(20));
        assert_eq!(queue.len(), 3);

        // Verify remaining order
        assert_eq!(queue.dequeue(), Some(10));
        assert_eq!(queue.dequeue(), Some(30));
        assert_eq!(queue.dequeue(), Some(40));

        // Try to remove non-existent call
        assert!(!queue.remove(999));
    }

    #[test]
    fn test_queue_iter() {
        let mut queue = Queue::new();

        queue.enqueue(5);
        queue.enqueue(10);
        queue.enqueue(15);

        let ids: Vec<_> = queue.iter().copied().collect();
        assert_eq!(ids, vec![5, 10, 15]);
    }
}
