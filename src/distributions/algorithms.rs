//! Algorithms for sampling from probability distributions

pub mod box_muller;
pub use box_muller::{box_muller_transform, scaled_box_muller_transform};

pub mod zignor;
pub use zignor::{zignor_method, scaled_zignor_method};
