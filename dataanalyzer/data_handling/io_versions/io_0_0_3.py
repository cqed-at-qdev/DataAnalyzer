import xarray as xr
from dataanalyzer.data_handling.utilities import (
    attrs_to_string,
    seperate_from_unumpy,
    combine_to_unumpy,
    string_to_attrs,
)


def save(ds: xr.Dataset, path: str) -> None:
    """Saves dataset as netcdf to path, version 0.0.3"""
    ds = attrs_to_string(ds, type(None))
    ds = seperate_from_unumpy(ds)
    ds.to_netcdf(path)


def format_loaded_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Formats the loaded dataset, version 0.0.3

        -Errors are combined with data as unumpy arrays

        -None strings are restored to None type
    """
    ds = string_to_attrs(ds, str(type(None)), lambda s: None)
    ds = combine_to_unumpy(ds)
    return ds
