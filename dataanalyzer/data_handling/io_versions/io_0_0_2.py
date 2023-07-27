import xarray as xr
from dataanalyzer.data_handling.utilities import attrs_to_string, string_to_attrs


def save(ds: xr.Dataset, path: str) -> None:
    """Saves dataset as netcdf to path, SQuID lab data_version 0.0.2"""
    ds = attrs_to_string(ds, type(None))
    ds.to_netcdf(path)


def format_loaded_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Formats the loaded dataset, SQuID lab data_version 0.0.2 \n
    None strings are restored to None type"""
    ds = string_to_attrs(ds, str(type(None)), lambda s: None)
    return ds
