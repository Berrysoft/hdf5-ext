//! DST extension APIs for [`hdf5`].

#![feature(ptr_metadata)]
#![cfg_attr(test, feature(maybe_uninit_write_slice))]
#![warn(missing_docs)]

mod ext;
pub use ext::*;

use dst_container::{UnsizedSlice, UnsizedStr};
use hdf5::{
    types::{CompoundField, CompoundType, TypeDescriptor},
    H5Type,
};
use std::alloc::Layout;

#[doc(hidden)]
pub mod __internal {
    pub use hdf5::types::{CompoundField, CompoundType, TypeDescriptor};
}

pub use hdf5_dst_derive::H5TypeUnsized;

/// An extension version of [`H5Type`] for DST.
pub trait H5TypeUnsized {
    /// Get the [`TypeDescriptor`] of current data.
    fn type_descriptor(&self) -> TypeDescriptor;
}

impl<T: H5Type> H5TypeUnsized for T {
    fn type_descriptor(&self) -> TypeDescriptor {
        Self::type_descriptor()
    }
}

impl<T: H5Type> H5TypeUnsized for [T] {
    fn type_descriptor(&self) -> TypeDescriptor {
        TypeDescriptor::FixedArray(Box::new(T::type_descriptor()), self.len())
    }
}

impl H5TypeUnsized for str {
    fn type_descriptor(&self) -> TypeDescriptor {
        TypeDescriptor::FixedUnicode(self.len())
    }
}

impl<H: H5Type, T: H5Type> H5TypeUnsized for UnsizedSlice<H, T> {
    fn type_descriptor(&self) -> TypeDescriptor {
        let mut fields = vec![];
        let layout = Layout::new::<()>();
        let new_layout = Layout::for_value(&self.header);
        let (layout, offset) = layout.extend(new_layout).unwrap();
        fields.push(CompoundField::new(
            "header",
            H5TypeUnsized::type_descriptor(&self.header),
            offset,
            0usize,
        ));
        let new_layout = Layout::for_value(&self.slice);
        let (layout, offset) = layout.extend(new_layout).unwrap();
        fields.push(CompoundField::new(
            "slice",
            H5TypeUnsized::type_descriptor(&self.slice),
            offset,
            1usize,
        ));
        let layout = layout.pad_to_align();
        debug_assert_eq!(layout, Layout::for_value(self));
        let ty = CompoundType {
            fields,
            size: layout.size(),
        };
        TypeDescriptor::Compound(ty)
    }
}

impl<H: H5Type> H5TypeUnsized for UnsizedStr<H> {
    fn type_descriptor(&self) -> TypeDescriptor {
        let mut fields = vec![];
        let layout = Layout::new::<()>();
        let new_layout = Layout::for_value(&self.header);
        let (layout, offset) = layout.extend(new_layout).unwrap();
        fields.push(CompoundField::new(
            "header",
            H5TypeUnsized::type_descriptor(&self.header),
            offset,
            0usize,
        ));
        let new_layout = Layout::for_value(&self.str);
        let (layout, offset) = layout.extend(new_layout).unwrap();
        fields.push(CompoundField::new(
            "str",
            H5TypeUnsized::type_descriptor(&self.str),
            offset,
            1usize,
        ));
        let layout = layout.pad_to_align();
        debug_assert_eq!(layout, Layout::for_value(self));
        let ty = CompoundType {
            fields,
            size: layout.size(),
        };
        TypeDescriptor::Compound(ty)
    }
}

#[cfg(test)]
mod test {
    use crate::*;
    use dst_container::*;

    #[derive(MaybeUninitProject, H5TypeUnsized)]
    #[repr(C)]
    struct Foo {
        field1: i32,
        field2: i64,
        slice: [f32],
    }

    #[test]
    fn layout() {
        let foo: Box<Foo> = unsafe { Box::<Foo>::new_zeroed_unsized(3).assume_init() };
        let td = foo.type_descriptor();
        let ty = if let TypeDescriptor::Compound(ty) = td {
            ty
        } else {
            unreachable!()
        };
        assert_eq!(ty.size, 32);
        assert_eq!(ty.fields[0].offset, 0);
        assert_eq!(ty.fields[1].offset, 8);
        assert_eq!(ty.fields[2].offset, 16);
    }
}
