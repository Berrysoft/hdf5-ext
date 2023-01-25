use dst_container::*;
use hdf5::{h5call, h5lock, h5try, Dataset, Datatype, Dimension, Location, PropertyList, Result};
use hdf5_hl_sys::h5pt::{
    H5PTappend, H5PTclose, H5PTcreate, H5PTget_dataset, H5PTget_type, H5PTopen,
};
use hdf5_sys::h5i::{
    hid_t,
    H5I_type_t::{self, H5I_BADID, H5I_NTYPES},
    H5Iget_type, H5Iinc_ref,
};
use std::{ffi::CString, fmt::Debug, mem::transmute};

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
    pub const fn id(&self) -> hid_t {
        self.0
    }

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
}

impl PacketTable {
    pub fn create(
        loc: &Location,
        table_name: impl AsRef<str>,
        dtype: &Datatype,
        chunk_size: impl Dimension,
        plist: &PropertyList,
    ) -> Result<Self> {
        let table_name = CString::new(table_name.as_ref()).map_err(|e| e.to_string())?;

        let table = h5try!(H5PTcreate(
            loc.id(),
            table_name.as_ptr(),
            dtype.id(),
            chunk_size.size() as _,
            plist.id()
        ));

        Ok(Self(table))
    }

    pub fn open(loc: &Location, dset_name: impl AsRef<str>) -> Result<Self> {
        let dset_name = CString::new(dset_name.as_ref()).map_err(|e| e.to_string())?;

        let table = h5try!(H5PTopen(loc.id(), dset_name.as_ptr()));

        Ok(Self(table))
    }

    pub fn push<T: ?Sized>(&self, val: &T) -> Result<()> {
        let (ptr, _) = (val as *const T).to_raw_parts();
        h5try!(H5PTappend(self.id(), 1, ptr as *const _));
        Ok(())
    }

    pub fn append<T>(&self, slice: &[T]) -> Result<()> {
        h5try!(H5PTappend(
            self.id(),
            slice.len(),
            slice.as_ptr() as *const _
        ));
        Ok(())
    }

    pub fn append_unsized<T: ?Sized>(&self, vec: &FixedVec<T>) -> Result<()> {
        let (ptr, _) = vec.as_ptr().to_raw_parts();
        h5try!(H5PTappend(self.id(), vec.len(), ptr as *const _));
        Ok(())
    }

    pub fn dataset(&self) -> Result<Dataset> {
        let dset = h5try!(H5PTget_dataset(self.id()));
        h5lock!(H5Iinc_ref(dset));
        Ok(unsafe { transmute(dset) })
    }

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

#[cfg(test)]
mod test {
    use crate::*;
    use hdf5::{plist::DatasetCreateBuilder, Datatype};
    use tempfile::NamedTempFile;

    #[test]
    fn append() {
        let file = NamedTempFile::new().unwrap();

        let vec = vec![1, 1, 4, 5, 1, 4];

        let data = hdf5::File::create(file.path()).unwrap();
        {
            let table = PacketTable::create(
                &data,
                "data",
                &Datatype::from_type::<i32>().unwrap(),
                16,
                &DatasetCreateBuilder::new().finish().unwrap(),
            )
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
