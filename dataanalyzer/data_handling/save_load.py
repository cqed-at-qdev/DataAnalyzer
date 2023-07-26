import xarray as xr
import importlib


def save_dataset(ds: xr.Dataset, path: str) -> None:
    """Saves dataset as netcdf to path"""
    data_version = ds.attrs["data_version"].replace(".", "_")
    save = importlib.import_module(
        f"dataanalyzer.data_handling.io_versions.io_{data_version}"
    ).save

    save(ds, path)


def load_dataset(path: str) -> xr.Dataset:
    """Loads dataset from path and formats it according to data_version"""
    ds = xr.open_dataset(path)
    data_version = ds.attrs["data_version"].replace(".", "_")
    format_loaded_dataset = importlib.import_module(
        f"dataanalyzer.data_handling.io_versions.io_{data_version}"
    ).format_loaded_dataset
    return format_loaded_dataset(ds)
