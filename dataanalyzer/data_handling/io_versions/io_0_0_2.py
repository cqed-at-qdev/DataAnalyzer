import xarray as xr
from dataanalyzer.data_handling.utilities import attrs_to_string, string_to_attrs


def save(ds: xr.Dataset, path: str) -> None:
    """Saves dataset as netcdf to path, version 0.0.2"""
    ds = attrs_to_string(ds, type(None))
    ds.to_netcdf(path)


def format_loaded_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Formats the loaded dataset, version 0.0.2
    No formats are needed in this version, so this function just returns the dataset"""
    ds = string_to_attrs(ds, str(type(None)), lambda s: None)
    return ds
