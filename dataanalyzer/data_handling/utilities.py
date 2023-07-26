import xarray as xr


def is_version_update_compatible(
    ds: xr.Dataset, old_version: str, new_version: str
) -> bool:
    """Checks whether the dataset is compatible with the given versions"""
    if "data_version" not in ds.attrs:
        print("Dataset has no data_version attribute")
        return False
    if ds.attrs["data_version"] == new_version:
        print("Dataset is already version {new_version}")
        return False
    if ds.attrs["data_version"] != old_version:
        print(
            f"Dataset is version {ds.attrs['data_version']}, but this function only supports version {old_version}"
        )
        return False
    return True
