use crate::H5TypeUnsized;
use dst_container::*;
use hdf5::{h5check, types::TypeDescriptor, Container, Datatype, Object, Result};
use hdf5_sys::{
    h5a::{H5Aread, H5Awrite},
    h5d::{H5Dread, H5Dwrite},
    h5i::H5I_type_t::H5I_ATTR,
    h5p::H5P_DEFAULT,
    h5s::H5S_ALL,
};

pub trait ContainerExt {
    fn write<T: ?Sized + H5TypeUnsized>(&self, v: &FixedVec<T>) -> Result<()>;

    fn write_scalar<T: ?Sized + H5TypeUnsized>(&self, val: &T) -> Result<()>;

    fn read<T: ?Sized + H5TypeUnsized>(&self, v: &mut FixedVec<T>) -> Result<()>;

    fn read_scalar<T: ?Sized + H5TypeUnsized + MaybeUninitProject>(
        &self,
        val: &mut T::Target,
    ) -> Result<()>;
}

fn is_attr(obj: &Object) -> bool {
    obj.id_type() == H5I_ATTR
}

fn write_container(c: &Container, mem_dtype: TypeDescriptor, buf: *const ()) -> Result<()> {
    let file_dtype = c.dtype()?;
    let mem_dtype = Datatype::from_descriptor(&mem_dtype)?;
    debug_assert_eq!(file_dtype, mem_dtype);

    let obj_id = c.id();
    let tp_id = mem_dtype.id();
    if is_attr(c) {
        h5check(unsafe { H5Awrite(obj_id, tp_id, buf.cast()) })?;
    } else {
        h5check(unsafe { H5Dwrite(obj_id, tp_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.cast()) })?;
    }
    Ok(())
}

fn read_container(c: &Container, mem_dtype: TypeDescriptor, buf: *mut ()) -> Result<()> {
    let file_dtype = c.dtype()?;
    let mem_dtype = Datatype::from_descriptor(&mem_dtype)?;
    debug_assert_eq!(file_dtype, mem_dtype);

    let obj_id = c.id();
    let tp_id = mem_dtype.id();
    if is_attr(c) {
        h5check(unsafe { H5Aread(obj_id, tp_id, buf.cast()) })?;
    } else {
        h5check(unsafe { H5Dread(obj_id, tp_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.cast()) })?;
    }
    Ok(())
}

impl ContainerExt for Container {
    fn write<T: ?Sized + H5TypeUnsized>(&self, v: &FixedVec<T>) -> Result<()> {
        debug_assert_eq!(self.ndim(), 0);
        debug_assert_eq!(self.shape()[0], v.len());
        let (ptr, _) = v.as_ptr().to_raw_parts();
        // SAFETY: only the metadata of the reference is used.
        write_container(self, unsafe { v.get_unchecked(0) }.type_descriptor(), ptr)
    }

    fn write_scalar<T: ?Sized + H5TypeUnsized>(&self, val: &T) -> Result<()> {
        debug_assert_eq!(self.ndim(), 0);
        let (ptr, _) = (val as *const T).to_raw_parts();
        write_container(self, val.type_descriptor(), ptr)
    }

    fn read<T: ?Sized + H5TypeUnsized>(&self, v: &mut FixedVec<T>) -> Result<()> {
        debug_assert_eq!(self.ndim(), 0);
        let new_len = self.shape()[0];
        v.reserve(new_len);
        let (ptr, _) = v.as_mut_ptr().to_raw_parts();
        // SAFETY: only the metadata of the reference is used.
        read_container(self, unsafe { v.get_unchecked(0) }.type_descriptor(), ptr)?;
        // SAFETY: read successfully.
        unsafe {
            v.set_len(v.len() + new_len);
        }
        Ok(())
    }

    fn read_scalar<T: ?Sized + H5TypeUnsized + MaybeUninitProject>(
        &self,
        val: &mut T::Target,
    ) -> Result<()> {
        debug_assert_eq!(self.ndim(), 0);
        let (ptr, metadata) = (val as *mut T::Target).to_raw_parts();
        // SAFETY: also done in dst-container
        let val_src: &T = unsafe { &*std::ptr::from_raw_parts(ptr, metadata) };
        read_container(self, val_src.type_descriptor(), ptr)
    }
}
