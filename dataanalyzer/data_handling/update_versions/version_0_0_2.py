"""
Updates dataset from version 0.0.2
Author: Jacob Hastrup
Created: 2023-07-26"""

import xarray as xr
from uncertainties import unumpy
from dataanalyzer.data_handling.utilities import (
    is_version_update_compatible,
)

PREVIOUS_VERSION = None
CURRENT_VERSION = "0.0.2"
NEXT_VERSION = "0.0.3"


def convert_up(ds: xr.Dataset) -> xr.Dataset:
    """Update dataset from version 0.0.2 to 0.0.3.
    Changes:
        - data errors are now stored with data as uarray instead of as separate data variable
    """

    if is_version_update_compatible(ds, CURRENT_VERSION, NEXT_VERSION):
        ds_updated = ds.copy()
        ds_updated = _errors_to_unumpy(ds_updated)
        ds_updated.attrs["data_version"] = NEXT_VERSION
    return ds_updated


def _errors_to_unumpy(ds: xr.Dataset) -> xr.Dataset:
    """Merge all data-error-pairs in the dataset into uarrays."""
    for key in ds.data_vars:
        if key.endswith("_error"):
            error_key = key
            var_key = (
                key[:-7] if key.endswith("__error") else key[:-6]
            )  # a typo caused some datasets to have tag _error instead of the intended __error

            ds[var_key].data = unumpy.uarray(
                nominal_values=ds[var_key], std_devs=ds[error_key]
            )
            ds = ds.drop(error_key)
    return ds
