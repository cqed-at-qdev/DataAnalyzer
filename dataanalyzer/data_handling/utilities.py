import xarray as xr


def attrs_to_string(ds: xr.Dataset, attr_type: type) -> xr.Dataset:
    """Replaces attrs of type attr_type with a string representation to allow saving to netcdf.
    For example, None type cannot be saved to netcdf, but str(None) can."""
    ds = ds.copy()

    # Replace attributes of coordinates
    for coord_key in ds.coords:
        for attr_key, attr_value in ds.coords[coord_key].attrs.items():
            if type(attr_value) == attr_type:
                ds.coords[coord_key].attrs[attr_key] = str(attr_value)

    # Replace attributes of dataarrays
    for da_key in ds:
        for attr_key, attr_value in ds[da_key].attrs.items():
            if type(attr_value) == attr_type:
                ds[da_key].attrs[attr_key] = str(attr_value)

    # Replace attributes of dimensions
    for dim_key in ds.dims:
        for attr_key, attr_value in ds[dim_key].attrs.items():
            if type(attr_value) == attr_type:
                ds[dim_key].attrs[attr_key] = str(attr_value)

    # Replace global attributes of dataset
    for attr_key, attr_value in ds.attrs.items():
        if type(attr_value) == attr_type:
            ds.attrs[attr_key] = str(attr_value)
    return ds


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


####################################################################################################
### unumpy handling functions #######################################################################
####################################################################################################


def seperate_from_unumpy(ds: xr.Dataset) -> xr.Dataset:
    """Seperate data wtih unumpy errors out to seperate dimension to enable dataset saving"""
    from uncertainties import unumpy

    ds = ds.copy()
    ds.coords["value_error"] = ["value", "error"]
    ds.coords["value_error"].attrs["description"] = "values and errors"

    for key in ds.data_vars:
        if _contains_unumpy(ds[key]):
            data = ds[key].data
            ds[key] = ds[key].expand_dims({"value_error": 2}).copy()
            ds[key].loc[{"value_error": "value"}] = unumpy.nominal_values(data)
            ds[key].loc[{"value_error": "error"}] = unumpy.std_devs(data)

    return ds


def combine_to_unumpy(ds: xr.Dataset) -> xr.Dataset:
    """Combines data with errors along seperate dimension to unumpy array"""
    from uncertainties import unumpy

    ds = ds.copy()
    if "value_error" not in ds.coords:
        return ds
    for key in ds.data_vars:
        if "value_error" in ds[key].dims:
            data_unumpy = unumpy.uarray(
                ds[key].loc[{"value_error": "value"}].data,
                ds[key].loc[{"value_error": "error"}].data,
            )
            ds[key] = ds[key].astype("object")
            ds[key].loc[{"value_error": "value"}] = data_unumpy

            # Remove error dimension from data array
            ds[key] = ds[key].sel({"value_error": "value"}, drop=True)

    ds = ds.drop_dims("value_error")
    return ds


def _contains_unumpy(da: xr.DataArray) -> bool:
    import uncertainties

    """Checks whether the data array contains unumpy values"""
    return isinstance(da.data.take(0), uncertainties.core.AffineScalarFunc)
