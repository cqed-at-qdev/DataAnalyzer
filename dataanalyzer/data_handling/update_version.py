import importlib
import xarray as xr


def update_to_latest(ds: xr.Dataset) -> xr.Dataset:
    """Update the dataset to latest version"""
    is_updated = False
    while not is_updated:
        current_version = ds.attrs["data_version"].replace(".", "_")
        update_module = importlib.import_module(
            name=f"dataanalyzer.data_handling.update_versions.version_{current_version}"
        )

        if update_module.NEXT_VERSION is None:
            is_updated = True
        else:
            ds = update_module.convert_up(ds)
    return ds


def update_to_version(ds: xr.Dataset, version: str) -> xr.Dataset:
    """Update the dataset to the given version

    Args:
        ds: Dataset to update
        version: Version to update to, format "x_x_x" or "x.x.x"
    """

    version = version.replace(".", "_")

    is_updated = False
    while not is_updated:
        current_version = ds.attrs["data_version"].replace(".", "_")
        if current_version == version:
            is_updated = True
            break

        update_module = importlib.import_module(
            name=f"dataanalyzer.data_handling.update_versions.version_{current_version}"
        )

        if current_version > version:
            ds = update_module.convert_down(ds)
        else:
            ds = update_module.convert_up(ds)

    return ds
