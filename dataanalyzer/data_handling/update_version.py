import importlib
import xarray as xr


def update_to_latest(ds: xr.Dataset) -> xr.Dataset:
    """Update the dataset to latest version"""
    is_updated = False
    while not is_updated:
        current_version = ds.attrs["data_version"].replace(".", "_")
        update_module_name = f"dataanalyzer.data_handling.update_from_{current_version}"
        if importlib.util.find_spec(update_module_name) is None:
            is_updated = True
        else:
            update_dataset = importlib.import_module(update_module_name).update_dataset
            ds = update_dataset(ds)
    return ds


s = importlib.import_module(
    "dataanalyzer.data_handling.update_from_0_0_2"
).update_dataset
