mod buf_writer;
pub use buf_writer::*;

use dst_container::*;
use hdf5::{
    h5call, h5lock, h5try, plist::DatasetCreate, types::TypeDescriptor, Dataset, Datatype,
    Dimension, Group, H5Type, Result,
};
use hdf5_dst::H5TypeUnsized;
use hdf5_hl_sys::h5pt::{
    H5PTappend, H5PTclose, H5PTcreate, H5PTcreate_index, H5PTget_dataset, H5PTget_index,
    H5PTget_next, H5PTget_num_packets, H5PTget_type, H5PTis_valid, H5PTis_varlen, H5PTopen,
    H5PTread_packets, H5PTset_index,
};
use hdf5_sys::{
    h5i::{
        hid_t,
        H5I_type_t::{self, H5I_BADID, H5I_NTYPES},
        H5Iget_type, H5Iinc_ref,
    },
    h5p::H5P_DEFAULT,
};
use std::{
    ffi::CString,
    fmt::Debug,
    mem::{transmute, MaybeUninit},
    ptr::Pointee,
};

/// The packet type of a packet table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum PacketTableType {
    /// Fixed length packet.
    Fixed = 0,
    /// Variable length packet.
    VarLen = 1,
}

/// The HDF5 Packet Table is designed to allow records to be appended to and read from a table.
/// Packet Table datasets are chunked, allowing them to grow as needed.
#[repr(transparent)]
pub struct PacketTable(hid_t);

// Object impls.
impl Debug for PacketTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<HDF5: packet table>")
    }
}

// Object impls.
impl PacketTable {
    #[doc(hidden)]
    pub const fn id(&self) -> hid_t {
        self.0
    }

    #[doc(hidden)]
    pub fn id_type(&self) -> H5I_type_t {
        if self.id() <= 0 {
            H5I_BADID
        } else {
            match h5lock!(H5Iget_type(self.id())) {
                tp if tp > H5I_BADID && tp < H5I_NTYPES => tp,
                _ => H5I_BADID,
            }
        }
    }

    pub(crate) fn from_id(id: hid_t) -> Self {
        Self(id)
    }
}

impl PacketTable {
    /// Create a packet table builder from a specified location.
    pub fn builder(loc: &Group) -> PacketTableBuilder {
        PacketTableBuilder::new(loc)
    }

    /// Open an existing packet table.
    pub fn open(loc: &Group, dset_name: impl AsRef<str>) -> Result<Self> {
        let dset_name = CString::new(dset_name.as_ref()).map_err(|e| e.to_string())?;
        let table = h5try!(H5PTopen(loc.id(), dset_name.as_ptr()));
        Ok(Self::from_id(table))
    }

    /// Push one element into the packet table.
    pub fn push<T: ?Sized>(&mut self, val: &T) -> Result<()> {
        let (ptr, _) = (val as *const T).to_raw_parts();
        h5try!(H5PTappend(self.id(), 1, ptr as *const _));
        Ok(())
    }

    /// Append a slice into the packet table.
    pub fn append<T>(&mut self, slice: &[T]) -> Result<()> {
        h5try!(H5PTappend(
            self.id(),
            slice.len(),
            slice.as_ptr() as *const _
        ));
        Ok(())
    }

    /// Append an unsized vector into the packet table.
    pub fn append_unsized<T: ?Sized>(&mut self, vec: &FixedVec<T>) -> Result<()> {
        let (ptr, _) = vec.as_ptr().to_raw_parts();
        h5try!(H5PTappend(self.id(), vec.len(), ptr as *const _));
        Ok(())
    }

    /// Get the inner [`Dataset`] from the packet table.
    pub fn dataset(&self) -> Result<Dataset> {
        let dset = h5try!(H5PTget_dataset(self.id()));
        h5lock!(H5Iinc_ref(dset));
        Ok(unsafe { transmute(dset) })
    }

    /// Determine if the current packet table is valid.
    pub fn validate(&self) -> Result<()> {
        h5try!(H5PTis_valid(self.id()));
        Ok(())
    }

    /// Determines whether a packet table contains variable-length or fixed-length packets.
    pub fn table_type(&self) -> Result<PacketTableType> {
        let ty = h5try!(H5PTis_varlen(self.id()));
        Ok(unsafe { transmute(ty) })
    }

    /// Get the inner [`Datatype`] from the packet table.
    pub fn dtype(&self) -> Result<Datatype> {
        let ty = h5try!(H5PTget_type(self.id()));
        h5lock!(H5Iinc_ref(ty));
        Ok(unsafe { transmute(ty) })
    }

    /// Get the number of packets.
    pub fn num_packets(&self) -> Result<u64> {
        let mut len = 0;
        h5try!(H5PTget_num_packets(self.id(), &mut len));
        Ok(len)
    }

    /// Reset the current index to 0.
    pub fn reset_index(&mut self) -> Result<()> {
        h5try!(H5PTcreate_index(self.id()));
        Ok(())
    }

    /// Set the current index.
    pub fn set_index(&mut self, index: u64) -> Result<()> {
        h5try!(H5PTset_index(self.id(), index));
        Ok(())
    }

    /// Get the current index.
    pub fn index(&self) -> Result<u64> {
        let mut index = 0;
        h5try!(H5PTget_index(self.id(), &mut index));
        Ok(index)
    }

    fn read_impl<T>(
        &self,
        len: usize,
        f: impl FnOnce(&mut [MaybeUninit<T>]) -> Result<()>,
    ) -> Result<Vec<T>> {
        let mut vec = Vec::with_capacity(len);
        let uninit = vec.spare_capacity_mut();
        f(uninit)?;
        // SAFETY: read succeeded.
        unsafe {
            vec.set_len(len);
        }
        Ok(vec)
    }

    /// Read from a specified packet index and take some data.
    pub fn read<T>(&self, start: u64, len: usize) -> Result<Vec<T>> {
        self.read_impl(len, |uninit| {
            h5try!(H5PTread_packets(
                self.id(),
                start,
                len,
                uninit.as_mut_ptr() as *mut _
            ));
            Ok(())
        })
    }

    /// Read from current index and update the index if the operation succeeds.
    pub fn read_next<T>(&mut self, len: usize) -> Result<Vec<T>> {
        self.read_impl(len, |uninit| {
            h5try!(H5PTget_next(self.id(), len, uninit.as_mut_ptr() as *mut _));
            Ok(())
        })
    }

    fn read_unsized_impl<T: ?Sized>(
        &self,
        len: usize,
        buffer: &mut FixedVec<T>,
        f: impl FnOnce(*mut ()) -> Result<()>,
    ) -> Result<()> {
        let old_len = buffer.len();
        buffer.reserve(len);
        let (ptr, _) = unsafe { buffer.get_unchecked_mut(old_len) as *mut T }.to_raw_parts();
        f(ptr)?;
        unsafe {
            buffer.set_len(old_len + len);
        }
        Ok(())
    }

    /// Read from a specified packet index and take some data.
    pub fn read_unsized<T: ?Sized>(
        &self,
        start: u64,
        len: usize,
        buffer: &mut FixedVec<T>,
    ) -> Result<()> {
        self.read_unsized_impl(len, buffer, |ptr| {
            h5try!(H5PTread_packets(self.id(), start, len, ptr as *mut _));
            Ok(())
        })
    }

    /// Read from current index and update the index if the operation succeeds.
    pub fn read_next_unsized<T: ?Sized>(
        &mut self,
        len: usize,
        buffer: &mut FixedVec<T>,
    ) -> Result<()> {
        self.read_unsized_impl(len, buffer, |ptr| {
            h5try!(H5PTget_next(self.id(), len, ptr as *mut _));
            Ok(())
        })
    }

    /// Create an iterator to read the packets one by one.
    /// It doesn't influence the index of the packet table.
    #[allow(clippy::needless_lifetimes)]
    pub fn iter<'a, T>(&'a self) -> impl Iterator<Item = Result<T>> + 'a {
        let mut index = 0u64;
        std::iter::from_fn::<Result<T>, _>(move || {
            let mut read_one = || {
                if index < self.num_packets()? {
                    let mut val = MaybeUninit::uninit();
                    h5try!(H5PTread_packets(
                        self.id(),
                        index,
                        1,
                        val.as_mut_ptr() as *mut _
                    ));
                    index += 1;
                    Ok(Some(unsafe { val.assume_init() }))
                } else {
                    Ok(None)
                }
            };
            read_one().transpose()
        })
    }
}

impl Drop for PacketTable {
    fn drop(&mut self) {
        h5call!(H5PTclose(self.id())).unwrap();
    }
}

/// The incomplete builder of [`PacketTable`].
/// You need at least set the chunk or the plist with valid chunk.
/// If both are set, the chunk value will override the plist chunk value.
pub struct PacketTableBuilder {
    loc: Group,
    chunk: Option<usize>,
    plist: Option<DatasetCreate>,
}

impl PacketTableBuilder {
    pub(crate) fn new(loc: &Group) -> Self {
        Self {
            loc: loc.clone(),
            chunk: None,
            plist: None,
        }
    }

    /// Set the [`DatasetCreate`] property list.
    pub fn plist(mut self, plist: DatasetCreate) -> Self {
        self.plist = Some(plist);
        self
    }

    /// Set the chunk size of the packet table.
    pub fn chunk(mut self, chunk: impl Dimension) -> Self {
        self.chunk = Some(chunk.size());
        self
    }

    /// Set the [`Datatype`] of the packet table.
    pub fn dtype<T: H5Type>(self) -> PacketTableBuilderTyped {
        PacketTableBuilderTyped {
            builder: self,
            dtype: T::type_descriptor(),
        }
    }

    /// Set the [`Datatype`] of the packet table with provided [`TypeDescriptor`].
    pub fn dtype_as(self, dtype: TypeDescriptor) -> PacketTableBuilderTyped {
        PacketTableBuilderTyped {
            builder: self,
            dtype,
        }
    }

    /// Set the [`Datatype`] of the packet table with raw pointee metadata.
    pub fn dtype_unsized<T: ?Sized + H5TypeUnsized>(
        self,
        metadata: <T as Pointee>::Metadata,
    ) -> PacketTableBuilderTyped {
        let ptr: *const T = std::ptr::from_raw_parts(std::ptr::null::<()>(), metadata);
        self.dtype_unsized_like(ptr)
    }

    /// Set the [`Datatype`] of the packet table with a sample value reference.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn dtype_unsized_like<T: ?Sized + H5TypeUnsized>(
        self,
        ptr: *const T,
    ) -> PacketTableBuilderTyped {
        let dtype = unsafe { (*ptr).type_descriptor() };
        PacketTableBuilderTyped {
            builder: self,
            dtype,
        }
    }

    pub(crate) fn create(self, table_name: &str, dtype: &Datatype) -> Result<PacketTable> {
        if let Some(plist) = &self.plist {
            if plist.chunk().is_none() {
                return Err("Invalid chunk.".into());
            }
        } else if self.chunk.is_none() {
            return Err("Either plist or chunk need to be set.".into());
        }
        let table_name = CString::new(table_name).map_err(|e| e.to_string())?;
        let plist = self
            .plist
            .as_ref()
            .map(|plist| plist.id())
            .unwrap_or(H5P_DEFAULT);
        let table = h5try!(H5PTcreate(
            self.loc.id(),
            table_name.as_ptr(),
            dtype.id(),
            self.chunk.unwrap_or_default() as _,
            plist
        ));
        Ok(PacketTable::from_id(table))
    }
}

/// A complete builder of [`PacketTable`].
pub struct PacketTableBuilderTyped {
    builder: PacketTableBuilder,
    dtype: TypeDescriptor,
}

impl PacketTableBuilderTyped {
    /// Set the [`DatasetCreate`] property list.
    pub fn plist(mut self, plist: DatasetCreate) -> Self {
        self.builder = self.builder.plist(plist);
        self
    }

    /// Set the chunk size of the packet table.
    pub fn chunk(mut self, chunk: impl Dimension) -> Self {
        self.builder = self.builder.chunk(chunk);
        self
    }

    /// Create the [`PacketTable`].
    pub fn create(self, table_name: impl AsRef<str>) -> Result<PacketTable> {
        let dtype = Datatype::from_descriptor(&self.dtype)?;
        self.builder.create(table_name.as_ref(), &dtype)
    }
}

#[cfg(test)]
mod test {
    use crate::*;
    use hdf5::types::VarLenArray;
    use tempfile::NamedTempFile;

    #[test]
    fn basic() {
        let file = NamedTempFile::new().unwrap();

        let vec = vec![1, 1, 4, 5, 1, 4];

        let data = hdf5::File::create(file.path()).unwrap();
        {
            let mut table = PacketTable::builder(&data)
                .chunk(16)
                .dtype::<i32>()
                .create("data")
                .unwrap();
            table.append(&vec).unwrap();
        }
        {
            let mut table = PacketTable::open(&data, "data").unwrap();
            {
                let dataset = table.dataset().unwrap();
                let read_data = dataset.read_1d::<i32>().unwrap();
                assert_eq!(read_data.as_slice().unwrap(), &[1, 1, 4, 5, 1, 4]);
            }
            {
                let read_data = table.read::<i32>(0, 6).unwrap();
                assert_eq!(read_data, &[1, 1, 4, 5, 1, 4]);
            }
            {
                let read_data = table
                    .iter::<i32>()
                    .map(|item| item.unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(read_data, &[1, 1, 4, 5, 1, 4]);
            }
            {
                table.reset_index().unwrap();
                assert_eq!(table.index().unwrap(), 0);
                assert_eq!(table.read_next::<i32>(6).unwrap(), &[1, 1, 4, 5, 1, 4]);
                assert_eq!(table.index().unwrap(), 6);
            }
        }
    }

    #[test]
    fn varlen() {
        let file = NamedTempFile::new().unwrap();

        let arr1 = VarLenArray::from_slice(&[1, 1, 4]);
        let arr2 = VarLenArray::from_slice(&[5, 1]);
        let arr3 = VarLenArray::from_slice(&[4]);
        let arr4 = VarLenArray::from_slice(&[1, 9, 1, 9]);

        let data = hdf5::File::create(file.path()).unwrap();
        {
            let mut table = PacketTable::builder(&data)
                .chunk(16)
                .dtype::<VarLenArray<i32>>()
                .create("data")
                .unwrap();
            table.push(&arr1).unwrap();
            table.push(&arr2).unwrap();
            table.push(&arr3).unwrap();
            table.push(&arr4).unwrap();
        }
        {
            let table = PacketTable::open(&data, "data").unwrap();

            let mut iter = table.iter::<VarLenArray<i32>>();
            assert_eq!(iter.next().unwrap().unwrap(), arr1);
            assert_eq!(iter.next().unwrap().unwrap(), arr2);
            assert_eq!(iter.next().unwrap().unwrap(), arr3);
            assert_eq!(iter.next().unwrap().unwrap(), arr4);
        }
    }
}

#[cfg(test)]
#[generic_tests::define(attrs(bench))]
mod bench_chunk {
    use crate::*;
    use tempfile::NamedTempFile;
    use test::Bencher;

    #[bench]
    fn chunk_buffer<const C: usize, const B: usize>(b: &mut Bencher) {
        let file = NamedTempFile::new().unwrap();
        let file = hdf5::File::create(file.path()).unwrap();
        let mut table = PacketTable::builder(&file)
            .chunk(C)
            .dtype::<i32>()
            .create("data")
            .unwrap();
        if B == 1 {
            b.iter(|| {
                for i in 0..65536 {
                    table.push(&i).unwrap();
                }
            })
        } else {
            b.iter(|| {
                let mut writer = PacketTableBufWriter::<i32>::new(&mut table, B);
                for i in 0..65536 {
                    writer.push(i).unwrap();
                }
            })
        }
    }

    #[instantiate_tests(<16, 1>)]
    mod append_16_1 {}

    #[instantiate_tests(<16, 16>)]
    mod append_16_16 {}

    #[instantiate_tests(<16, 1024>)]
    mod append_16_1024 {}

    #[instantiate_tests(<16, 65536>)]
    mod append_16_65536 {}

    #[instantiate_tests(<1024, 1>)]
    mod append_1024_1 {}

    #[instantiate_tests(<1024, 16>)]
    mod append_1024_16 {}

    #[instantiate_tests(<1024, 1024>)]
    mod append_1024_1024 {}

    #[instantiate_tests(<1024, 65536>)]
    mod append_1024_65536 {}

    #[instantiate_tests(<65536, 1>)]
    mod append_65536_1 {}

    #[instantiate_tests(<65536, 16>)]
    mod append_65536_16 {}

    #[instantiate_tests(<65536, 1024>)]
    mod append_65536_1024 {}

    #[instantiate_tests(<65536, 65536>)]
    mod append_65536_65536 {}
}
