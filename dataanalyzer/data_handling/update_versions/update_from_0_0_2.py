"""
Updates dataset from version 0.0.2 to 0.0.3
To update a dataset, ds, run the update_dataset(ds) function
Author: Jacob Hastrup
Created: 2023-07-26"""

import xarray as xr
from uncertainties import unumpy
from dataanalyzer.data_handling.utilities import is_version_update_compatible

OLD_VERSION = "0.0.2"
NEW_VERSION = "0.0.3"


def update_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Update dataset from version 0.0.2 to 0.0.3
    Changes:
        - data error is now stored with data as uarray instead of as separate data variable
    """

    if is_version_update_compatible(ds, OLD_VERSION, NEW_VERSION):
        ds_updated = ds.copy()
        ds_updated = _errors_to_unumpy(ds_updated)
        ds_updated.attrs["data_version"] = NEW_VERSION
    return ds_updated


def _errors_to_unumpy(ds: xr.Dataset) -> xr.Dataset:
    """Merge data and errors into uarray"""
    for key in ds.data_vars:
        if key.endswith("__error"):
            error_key = key
            var_key = key[:-7]
            ds[var_key].data = unumpy.uarray(nominal_values=ds[var_key], std_devs=ds[error_key])
            ds = ds.drop(error_key)
    return ds
