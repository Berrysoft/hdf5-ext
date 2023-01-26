use crate::*;
use dst_container::{FixedVec, UnsizedClone};
use hdf5::Result;
use std::ptr::Pointee;

/// A [`PacketTable`] writer with buffer.
pub struct PacketTableBufWriter<'a, T: ?Sized> {
    table: &'a mut PacketTable,
    buffer: FixedVec<T>,
    buf_len: usize,
}

impl<'a, T: ?Sized> PacketTableBufWriter<'a, T> {
    /// Create a new [`PacketTableBufWriter`] with metadata and buffer length.
    pub fn new_unsized(
        table: &'a mut PacketTable,
        metadata: <T as Pointee>::Metadata,
        buf_len: usize,
    ) -> Self {
        Self {
            table,
            buffer: FixedVec::with_capacity(metadata, buf_len),
            buf_len,
        }
    }

    /// Create a new [`PacketTableBufWriter`] with metadata and buffer length.
    /// The metadata is obtained from the provided pointer.
    pub fn new_unsized_like(table: &'a mut PacketTable, ptr: *const T, buf_len: usize) -> Self {
        Self {
            table,
            buffer: FixedVec::with_capacity_like(ptr, buf_len),
            buf_len,
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.table.append_unsized(&self.buffer)?;
        self.buffer.clear();
        Ok(())
    }

    fn check_and_flush(&mut self) -> Result<()> {
        if self.buffer.len() >= self.buf_len {
            self.flush()?;
        }
        Ok(())
    }

    /// Clone the value into the buffer.
    pub fn push_clone(&mut self, val: &T) -> Result<()>
    where
        T: UnsizedClone,
    {
        self.buffer.push_clone(val);
        self.check_and_flush()
    }
}

impl<'a, T> PacketTableBufWriter<'a, T> {
    /// Create a new [`PacketTableBufWriter`] with buffer length.
    pub fn new(table: &'a mut PacketTable, buf_len: usize) -> Self {
        Self {
            table,
            buffer: FixedVec::with_capacity((), buf_len),
            buf_len,
        }
    }

    /// Push the value into the buffer.
    pub fn push(&mut self, val: T) -> Result<()> {
        unsafe {
            self.buffer.push_with(|uninit| {
                uninit.write(val);
            });
        }
        self.check_and_flush()
    }
}

impl<'a, T: ?Sized> Drop for PacketTableBufWriter<'a, T> {
    fn drop(&mut self) {
        if !self.buffer.is_empty() {
            self.flush().unwrap();
        }
    }
}
