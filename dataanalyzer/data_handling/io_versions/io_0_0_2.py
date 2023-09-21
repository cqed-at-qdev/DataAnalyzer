import xarray as xr
from dataanalyzer.data_handling.utilities import format_none_type, restore_none_type


def save(ds: xr.Dataset, path: str) -> None:
    """Saves dataset as netcdf to path, SQuID lab data_version 0.0.2"""
    ds = format_none_type(ds)
    ds.to_netcdf(path)


def format_loaded_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Formats the loaded dataset, SQuID lab data_version 0.0.2 \n
    None strings are restored to None type"""
    ds = restore_none_type(ds)
    return ds
