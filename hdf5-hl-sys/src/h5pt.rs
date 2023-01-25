use core::ffi::{c_char, c_int, c_size_t, c_void};
use hdf5_sys::{
    h5::{herr_t, hsize_t},
    h5i::hid_t,
};

extern "C" {
    pub fn H5PTcreate(
        loc_id: hid_t,
        ptable_name: *const c_char,
        dtype_id: hid_t,
        chunk_size: hsize_t,
        plist_id: hid_t,
    ) -> hid_t;
    #[deprecated(note = "deprecated in HDF5 1.10.0, use H5PTcreate")]
    pub fn H5PTcreate_fl(
        loc_id: hid_t,
        ptable_name: *const c_char,
        dtype_id: hid_t,
        chunk_size: hsize_t,
        compression: c_int,
    ) -> hid_t;
    pub fn H5PTopen(loc_id: hid_t, dset_name: *const c_char) -> hid_t;
    pub fn H5PTclose(table_id: hid_t) -> herr_t;

    pub fn H5PTappend(table_id: hid_t, nrecords: c_size_t, data: *const c_void) -> herr_t;

    pub fn H5PTcreate_index(table_id: hid_t) -> herr_t;
    pub fn H5PTset_index(table_id: hid_t, pt_index: hsize_t) -> herr_t;

    pub fn H5PTread_packets(
        table_id: hid_t,
        start: hsize_t,
        nrecords: c_size_t,
        data: *mut c_void,
    ) -> herr_t;
    pub fn H5PTget_next(table_id: hid_t, nrecords: c_size_t, data: *mut c_void) -> herr_t;
    pub fn H5PTget_dataset(table_id: hid_t) -> hid_t;
    pub fn H5PTget_type(table_id: hid_t) -> hid_t;

    pub fn H5PTget_num_packets(table_id: hid_t, nrecords: *mut c_size_t) -> herr_t;
    pub fn H5PTis_valid(table_id: hid_t) -> herr_t;
    pub fn H5PTis_varlen(table_id: hid_t) -> herr_t;

    pub fn H5PTfree_vlen_buff(table_id: hid_t, bufflen: hsize_t, buff: *mut c_void) -> herr_t;
}
