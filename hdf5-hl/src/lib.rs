//! HDF5 High-level APIs for Rust.

#![feature(ptr_metadata)]
#![cfg_attr(test, feature(test))]
#![warn(missing_docs)]

#[cfg(test)]
extern crate test;

mod pt;
pub use pt::*;
