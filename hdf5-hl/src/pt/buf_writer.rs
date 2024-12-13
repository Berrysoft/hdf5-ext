use crate::*;
use dst_container::*;
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

    /// Force flush the buffer.
    pub fn flush(&mut self) -> Result<()> {
        if !self.buffer.is_empty() {
            self.table.append_unsized(&self.buffer)?;
            self.buffer.clear();
        }
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

    /// Push the value into the buffer.
    ///
    /// # Safety
    ///
    /// See [`FixedVec::push_with`].
    pub unsafe fn push_with(&mut self, f: impl FnOnce(&mut T::Target)) -> Result<()>
    where
        T: MaybeUninitProject,
    {
        unsafe { self.buffer.push_with(f) };
        self.check_and_flush()
    }
}

impl<'a, T> PacketTableBufWriter<'a, T> {
    /// Create a new [`PacketTableBufWriter`] with buffer length.
    pub fn new(table: &'a mut PacketTable, buf_len: usize) -> Self {
        Self::new_unsized(table, (), buf_len)
    }

    /// Push the value into the buffer.
    pub fn push(&mut self, val: T) -> Result<()> {
        // SAFETY: we are sure the value is initialized.
        unsafe {
            self.push_with(|uninit| {
                uninit.write(val);
            })
        }
    }
}

impl<T: ?Sized> Drop for PacketTableBufWriter<'_, T> {
    fn drop(&mut self) {
        self.flush().unwrap();
    }
}
