"""
Updates dataset from version 0.0.3
Author: Jacob Hastrup
Created: 2023-07-26"""

import xarray as xr
from uncertainties import unumpy
from dataanalyzer.data_handling.utilities import (
    is_version_update_compatible,
    contains_unumpy,
)

PREVIOUS_VERSION = "0.0.2"
CURRENT_VERSION = "0.0.3"
NEXT_VERSION = None


def convert_down(ds: xr.Dataset) -> xr.Dataset:
    """Back-converts dataset from version 0.0.3 to 0.0.2
    Changes:
        - data error is now stored as separate data variable instead of with data as uarray
    """

    if is_version_update_compatible(ds, CURRENT_VERSION, PREVIOUS_VERSION):
        ds_updated = ds.copy()
        ds_updated = _unumpy_to_errors(ds_updated)
        ds_updated.attrs["data_version"] = PREVIOUS_VERSION
    return ds_updated


def _unumpy_to_errors(ds: xr.Dataset) -> xr.Dataset:
    """Split all entries in dataset with uarray as data into data and errors"""
    for key in ds.data_vars:
        if contains_unumpy(ds.data_vars[key]):
            var_key = key
            error_key = key + "__error"
            ds[error_key] = ds[var_key].copy()
            ds[error_key].data = unumpy.std_devs(ds[var_key].data)
            ds[var_key].data = unumpy.nominal_values(ds[var_key].data)

    return ds
