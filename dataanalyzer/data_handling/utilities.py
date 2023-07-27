import xarray as xr
from typing import Protocol, Callable, Any


####################################################################################################
### handling of non-compatible types in attributes #################################################
####################################################################################################


# Typing class specifying that the object has an attrs attribute of type dict[str, any]
class HasAttrs(Protocol):
    attrs: dict[str, Any]


def attrs_to_string(ds: xr.Dataset, attr_type: type) -> xr.Dataset:
    """Replaces attrs of type attr_type with a string representation to allow saving to netcdf. Also adds a f"{key}__type" attribute to the attrs dict to allow restoring the original type.
    Intented for simple datatypes. For example, None type cannot be saved to netcdf, but str(None) can.
    Returns a copy of the dataset with the changed attributes"""

    def _replace_in_data(data: HasAttrs) -> None:
        to_be_added = {}
        for attr_key, attr_value in data.attrs.items():
            if type(attr_value) == attr_type:
                data.attrs[attr_key] = str(attr_value)
                to_be_added[f"{attr_key}__type"] = str(type(attr_value))
        data.attrs.update(to_be_added)

    ds = ds.copy()

    # Replace attributes of coordinates
    for coord_key in ds.coords:
        _replace_in_data(ds.coords[coord_key])

    # Replace attributes of dataarrays
    for da_key in ds:
        _replace_in_data(ds[da_key])

    # Replace attributes of dimensions
    for dim_key in ds.dims:
        _replace_in_data(ds[dim_key])

    # Replace global attributes of dataset
    _replace_in_data(ds)

    return ds


def string_to_attrs(
    ds: xr.Dataset, type_string: str, str_to_type_fnc: Callable[[str], Any]
) -> xr.Dataset:
    """Replaces attrs of type type_string with the result of str_to_type_fnc.
    Used to restore attributes that were replaced by attrs_to_string."""

    def _restore_attr(data: HasAttrs) -> None:
        to_be_deleted = []
        for attr_key, attr_value in data.attrs.items():
            if attr_key.endswith("__type") and attr_value == type_string:
                attr_main_key = attr_key[:-6]
                data.attrs[attr_main_key] = str_to_type_fnc(data.attrs[attr_main_key])
                to_be_deleted.append(attr_key)
        for key in to_be_deleted:
            del data.attrs[key]

    # Restore attributes of coordinates
    for coord_key in ds.coords:
        _restore_attr(ds.coords[coord_key])

    # Restore attributes of dataarrays
    for da_key in ds:
        _restore_attr(ds[da_key])

    # Restore attributes of dimensions
    for dim_key in ds.dims:
        _restore_attr(ds[dim_key])

    # Restore global attributes of dataset
    _restore_attr(ds)

    return ds


####################################################################################################
### Version checking ###############################################################################
####################################################################################################


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
### unumpy handling functions ######################################################################
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
