use hdf5::{types::TypeDescriptor, H5Type};

#[doc(hidden)]
pub mod __internal {
    pub use hdf5::types::{CompoundField, CompoundType, TypeDescriptor};
}

pub use hdf5_dst_derive::H5TypeUnsized;

pub trait H5TypeUnsized {
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
