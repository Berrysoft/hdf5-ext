use dst_container::*;
use hdf5::{
    h5call, h5lock, h5try, plist::DatasetCreate, types::TypeDescriptor, Dataset, Datatype,
    Dimension, Group, H5Type, Result,
};
use hdf5_dst::H5TypeUnsized;
use hdf5_hl_sys::h5pt::{
    H5PTappend, H5PTclose, H5PTcreate, H5PTget_dataset, H5PTget_type, H5PTopen,
};
use hdf5_sys::{
    h5i::{
        hid_t,
        H5I_type_t::{self, H5I_BADID, H5I_NTYPES},
        H5Iget_type, H5Iinc_ref,
    },
    h5p::H5P_DEFAULT,
};
use std::{ffi::CString, fmt::Debug, mem::transmute, ptr::Pointee};

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
    pub fn push<T: ?Sized>(&self, val: &T) -> Result<()> {
        let (ptr, _) = (val as *const T).to_raw_parts();
        h5try!(H5PTappend(self.id(), 1, ptr as *const _));
        Ok(())
    }

    /// Append a slice into the packet table.
    pub fn append<T>(&self, slice: &[T]) -> Result<()> {
        h5try!(H5PTappend(
            self.id(),
            slice.len(),
            slice.as_ptr() as *const _
        ));
        Ok(())
    }

    /// Append an unsized vector into the packet table.
    pub fn append_unsized<T: ?Sized>(&self, vec: &FixedVec<T>) -> Result<()> {
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

    /// Get the inner [`Datatype`] from the packet table.
    pub fn dtype(&self) -> Result<Datatype> {
        let ty = h5try!(H5PTget_type(self.id()));
        h5lock!(H5Iinc_ref(ty));
        Ok(unsafe { transmute(ty) })
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
        unsafe {
            let ptr: *const T = std::ptr::from_raw_parts(std::ptr::null(), metadata);
            self.dtype_unsized_like(&*ptr)
        }
    }

    /// Set the [`Datatype`] of the packet table with a sample value reference.
    pub fn dtype_unsized_like<T: ?Sized + H5TypeUnsized>(self, val: &T) -> PacketTableBuilderTyped {
        let dtype = val.type_descriptor();
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
    use tempfile::NamedTempFile;

    #[test]
    fn append() {
        let file = NamedTempFile::new().unwrap();

        let vec = vec![1, 1, 4, 5, 1, 4];

        let data = hdf5::File::create(file.path()).unwrap();
        {
            let table = PacketTable::builder(&data)
                .chunk(16)
                .dtype::<i32>()
                .create("data")
                .unwrap();
            table.append(&vec).unwrap();
        }
        {
            let table = PacketTable::open(&data, "data").unwrap();
            let dataset = table.dataset().unwrap();
            let read_data = dataset.read_1d::<i32>().unwrap();
            assert_eq!(read_data.as_slice().unwrap(), &[1, 1, 4, 5, 1, 4]);
        }
    }
}
