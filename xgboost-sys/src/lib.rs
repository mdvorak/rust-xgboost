#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unnecessary_transmutes)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    /// Read the last error XGBoost recorded, for assertion messages.
    fn last_error() -> String {
        unsafe { std::ffi::CStr::from_ptr(XGBGetLastError()) }
            .to_string_lossy()
            .into_owned()
    }

    /// The linked libxgboost must be the version this crate says it bundles.
    ///
    /// build.rs compares CARGO_PKG_VERSION against the submodule's
    /// version_config.h, but that only checks the *headers* bindgen read. It
    /// cannot catch cargo handing us a stale cached libxgboost from a previous
    /// pin, which has happened here before and surfaced far from the cause (a
    /// version-skewed library showed up as "Unknown objective function").
    /// Asking the loaded library itself is the only check that covers the
    /// artifact actually linked.
    ///
    /// Note this is deliberately stricter than build.rs, which only warns so that
    /// an intentional skew (a pre-release submodule pin, or a user-supplied
    /// XGBOOST_LIB_DIR) still *builds*. Building against a skewed library stays
    /// allowed; this crate's own test suite just does not claim to pass against
    /// one, since the pinned numerics elsewhere assume the bundled version.
    #[test]
    fn linked_library_version_matches_crate_version() {
        let (mut major, mut minor, mut patch) = (0, 0, 0);
        unsafe { XGBoostVersion(&mut major, &mut minor, &mut patch) };
        let linked = format!("{major}.{minor}.{patch}");
        assert_eq!(
            linked,
            env!("CARGO_PKG_VERSION"),
            "linked libxgboost is {linked} but this crate declares \
             {}; cargo may be reusing a library built from an older submodule pin",
            env!("CARGO_PKG_VERSION")
        );
    }

    #[test]
    fn read_matrix() {
        // `XGDMatrixCreateFromURI` takes a JSON config, not a bare path, and the
        // `?format=` query is required for text input. The path is relative to
        // this package's root, which is the cwd for its own test binary.
        let config = cr#"{"uri": "xgboost/demo/data/agaricus.txt.train?format=libsvm", "silent": 1}"#;

        let mut handle = std::ptr::null_mut();
        let ret_val = unsafe { XGDMatrixCreateFromURI(config.as_ptr(), &mut handle) };
        assert_eq!(ret_val, 0, "XGDMatrixCreateFromURI failed: {}", last_error());

        let mut num_rows = 0;
        let ret_val = unsafe { XGDMatrixNumRow(handle, &mut num_rows) };
        assert_eq!(ret_val, 0);
        assert_eq!(num_rows, 6513);

        let mut num_cols = 0;
        let ret_val = unsafe { XGDMatrixNumCol(handle, &mut num_cols) };
        assert_eq!(ret_val, 0);
        // 127, not the 126 this test asserted while it was never compiled: the
        // libsvm indices run to 126 and column 0 is kept, which is what
        // `DMatrix::load` sees too (src/dmatrix.rs `read_num_rows_cols`).
        assert_eq!(num_cols, 127);

        let ret_val = unsafe { XGDMatrixFree(handle) };
        assert_eq!(ret_val, 0);
    }
}
