//! Rust bindings to `hdf5_hl`.

#![feature(c_size_t)]

#[link(name = "hdf5_hl")]
unsafe extern "C" {}

pub mod h5pt;
