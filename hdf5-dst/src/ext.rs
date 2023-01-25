use std::ptr::Pointee;

use crate::H5TypeUnsized;
use dst_container::*;
use hdf5::{
    h5try, types::TypeDescriptor, AttributeBuilder, AttributeBuilderEmpty, Container,
    DatasetBuilder, DatasetBuilderEmpty, Datatype, Object, Result,
};
use hdf5_sys::{
    h5a::{H5Aread, H5Awrite},
    h5d::{H5Dread, H5Dwrite},
    h5i::H5I_type_t::H5I_ATTR,
    h5p::H5P_DEFAULT,
    h5s::H5S_ALL,
};

/// DST extensions for [`Container`].
pub trait ContainerExt {
    /// Writes a 1-dimensional array view into a dataset/attribute in memory order.
    ///
    /// The number of elements in the view must match the number of elements in the
    /// destination dataset/attribute. The input argument must be convertible to a
    /// 1-dimensional array view (this includes slices).
    fn write_unsized<T: ?Sized + H5TypeUnsized>(&self, v: &FixedVec<T>) -> Result<()>;

    /// Writes a scalar dataset/attribute.
    fn write_scalar_unsized<T: ?Sized + H5TypeUnsized>(&self, val: &T) -> Result<()>;

    /// Reads a dataset/attribute into a 1-dimensional array.
    ///
    /// The dataset/attribute must be 1-dimensional.
    fn read_unsized<T: ?Sized + H5TypeUnsized>(&self, v: &mut FixedVec<T>) -> Result<()>;

    /// Reads a scalar dataset/attribute.
    fn read_scalar_unsized<T: ?Sized + H5TypeUnsized + MaybeUninitProject>(
        &self,
        val: &mut T::Target,
    ) -> Result<()>;
}

/// Determine if the container is attribute.
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
        h5try!(H5Awrite(obj_id, tp_id, buf.cast()));
    } else {
        h5try!(H5Dwrite(
            obj_id,
            tp_id,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            buf.cast()
        ));
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
        h5try!(H5Aread(obj_id, tp_id, buf.cast()));
    } else {
        h5try!(H5Dread(
            obj_id,
            tp_id,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            buf.cast()
        ));
    }
    Ok(())
}

impl ContainerExt for Container {
    fn write_unsized<T: ?Sized + H5TypeUnsized>(&self, v: &FixedVec<T>) -> Result<()> {
        debug_assert_eq!(self.ndim(), 1);
        debug_assert_eq!(self.shape()[0], v.len());
        let (ptr, _) = v.as_ptr().to_raw_parts();
        // SAFETY: only the metadata of the reference is used.
        write_container(self, unsafe { v.get_unchecked(0) }.type_descriptor(), ptr)
    }

    fn write_scalar_unsized<T: ?Sized + H5TypeUnsized>(&self, val: &T) -> Result<()> {
        debug_assert_eq!(self.ndim(), 0);
        let (ptr, _) = (val as *const T).to_raw_parts();
        write_container(self, val.type_descriptor(), ptr)
    }

    fn read_unsized<T: ?Sized + H5TypeUnsized>(&self, v: &mut FixedVec<T>) -> Result<()> {
        debug_assert_eq!(self.ndim(), 1);
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

    fn read_scalar_unsized<T: ?Sized + H5TypeUnsized + MaybeUninitProject>(
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

/// DST extensions for [`DatasetBuilder`] and [`AttributeBuilder`].
pub trait ContainerBuilderExt: Sized {
    /// The empty builder type.
    type EmptyBuilder: Sized;

    /// DST version of [`empty`].
    /// Need pointee metadata passed in.
    fn empty_unsized<T: ?Sized + H5TypeUnsized>(
        self,
        metadata: <T as Pointee>::Metadata,
    ) -> Self::EmptyBuilder;

    /// Give a sample reference of DST instance, and create a dataset like it.
    fn empty_like_unsized<T: ?Sized + H5TypeUnsized>(self, val: &T) -> Self::EmptyBuilder {
        let (_, metadata) = (val as *const T).to_raw_parts();
        self.empty_unsized::<T>(metadata)
    }
}

fn type_from_null<T: ?Sized + H5TypeUnsized>(metadata: <T as Pointee>::Metadata) -> TypeDescriptor {
    // SAFETY: is it safe?
    unsafe {
        let ptr: *const T = std::ptr::from_raw_parts(std::ptr::null(), metadata);
        (*ptr).type_descriptor()
    }
}

impl ContainerBuilderExt for DatasetBuilder {
    type EmptyBuilder = DatasetBuilderEmpty;

    fn empty_unsized<T: ?Sized + H5TypeUnsized>(
        self,
        metadata: <T as Pointee>::Metadata,
    ) -> Self::EmptyBuilder {
        let ty = type_from_null::<T>(metadata);
        self.empty_as(&ty)
    }
}

impl ContainerBuilderExt for AttributeBuilder {
    type EmptyBuilder = AttributeBuilderEmpty;

    fn empty_unsized<T: ?Sized + H5TypeUnsized>(
        self,
        metadata: <T as Pointee>::Metadata,
    ) -> Self::EmptyBuilder {
        let ty = type_from_null::<T>(metadata);
        self.empty_as(&ty)
    }
}

#[cfg(test)]
mod test {
    use crate::*;
    use dst_container::*;
    use std::mem::MaybeUninit;
    use tempfile::NamedTempFile;

    #[derive(MaybeUninitProject, H5TypeUnsized)]
    #[repr(C)]
    struct Data {
        header: u32,
        slice: [u64],
    }

    #[test]
    fn dataset() {
        let file = NamedTempFile::new().unwrap();

        let mut vec: FixedVec<Data> = FixedVec::new(6);
        assert_eq!(vec.len(), 0);
        unsafe {
            vec.push_with(|slice| {
                slice.header.write(114514);
                MaybeUninit::write_slice(&mut slice.slice, &[1, 1, 4, 5, 1, 4]);
            })
        };

        let data = hdf5::File::create(file.path()).unwrap();
        {
            let dataset = data
                .new_dataset_builder()
                .empty_like_unsized(&vec[0])
                .shape(vec.len())
                .create("data")
                .unwrap();
            assert_eq!(&dataset.shape(), &[vec.len()]);
            dataset.write_unsized(&vec).unwrap();
        }
        {
            let mut read_vec: FixedVec<Data> = FixedVec::new(6);
            let dataset = data.dataset("data").unwrap();
            assert_eq!(&dataset.shape(), &[vec.len()]);
            dataset.read_unsized(&mut read_vec).unwrap();
            assert_eq!(read_vec[0].header, 114514);
            assert_eq!(&read_vec[0].slice, &[1, 1, 4, 5, 1, 4]);
        }
    }

    #[test]
    fn attribute() {
        let file = NamedTempFile::new().unwrap();

        let unsized_data: Box<Data> = unsafe {
            Box::<Data>::new_unsized_with(6, |slice| {
                slice.header.write(114514);
                MaybeUninit::write_slice(&mut slice.slice, &[1, 1, 4, 5, 1, 4]);
            })
        };

        let data = hdf5::File::create(file.path()).unwrap();
        let dataset = data.new_dataset::<i32>().create("data").unwrap();
        dataset.write_scalar(&114514).unwrap();
        {
            let attr = dataset
                .new_attr_builder()
                .empty_like_unsized(unsized_data.as_ref())
                .create("attr")
                .unwrap();
            attr.write_scalar_unsized(unsized_data.as_ref()).unwrap();
        }
        {
            let mut read_data: Box<<Data as MaybeUninitProject>::Target> =
                Box::<Data>::new_uninit_unsized(6);
            let attr = dataset.attr("attr").unwrap();
            attr.read_scalar_unsized::<Data>(read_data.as_mut())
                .unwrap();
            let read_data: Box<Data> = unsafe { read_data.assume_init() };
            assert_eq!(read_data.header, 114514);
            assert_eq!(&read_data.slice, &[1, 1, 4, 5, 1, 4]);
        }
    }
}
