import xarray as xr
from dataanalyzer.data_handling.utilities import (
    seperate_from_unumpy,
    combine_to_unumpy,
    flatten_attrs,
    unflatten_attrs,
    format_none_type,
    restore_none_type,
    format_ufloat_type,
    restore_ufloat_type,
)


def save(ds: xr.Dataset, path: str) -> None:
    """Saves dataset as netcdf to path, version 0.0.3"""
    ds = flatten_attrs(ds)
    ds = format_none_type(ds)
    ds = format_ufloat_type(ds)
    ds = seperate_from_unumpy(ds)

    ds.to_netcdf(path)


def format_loaded_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Formats the loaded dataset, version 0.0.3
        -Errors are combined with data as unumpy arrays
        -None strings are restored to None type
        -Attributes are unflattened
    """
    ds = combine_to_unumpy(ds)
    ds = restore_none_type(ds)
    ds = restore_ufloat_type(ds)
    ds = unflatten_attrs(ds)

    return ds
