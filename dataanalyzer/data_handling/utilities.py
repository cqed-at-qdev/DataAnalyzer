import xarray as xr
from typing import Protocol, Callable, Any


####################################################################################################
### handling of non-compatible types in attributes #################################################
####################################################################################################


# Typing class specifying that the object has an attrs attribute of type dict[str, any]
class HasAttrs(Protocol):
    attrs: dict[str, Any]


def format_all_attrs(ds: xr.Dataset, formatting_fnc: Callable[[HasAttrs], None]) -> xr.Dataset:
    """Formats the attributes of the dataset, including attrs of coords, data variables, dimensions, 
    and global attrs, using the given formatting function"""
    ds = ds.copy()

    # Replace attributes of coordinates
    for coord_key in ds.coords:
        formatting_fnc(ds.coords[coord_key])

    # Replace attributes of dataarrays
    for da_key in ds:
        formatting_fnc(ds[da_key])

    # Replace attributes of dimensions    
    for dim_key in ds.dims:
        formatting_fnc(ds[dim_key])

    # Replace global attributes of dataset
    formatting_fnc(ds)
    return ds


def attrs_to_string(ds: xr.Dataset, attr_type: type) -> xr.Dataset:
    """Replaces attrs of type attr_type with a string representation to allow saving to netcdf. Also adds a f"{key}__type" attribute to the attrs dict to allow restoring the original type.
    Intented for simple datatypes. For example, None type cannot be saved to netcdf, but str(None) can.
    Returns a copy of the dataset with the changed attributes"""

    def _replace_type_with_str(data: HasAttrs) -> None:
        type_identifiers = {}
        for attr_key, attr_value in data.attrs.items():
            if type(attr_value) == attr_type:
                data.attrs[attr_key] = str(attr_value)
                type_identifiers[f"{attr_key}__type"] = str(type(attr_value))
        data.attrs.update(type_identifiers)

    ds = format_all_attrs(ds, _replace_type_with_str)
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

    ds = format_all_attrs(ds, _restore_attr)

    return ds


def flatten_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Flattens the attributes of the dataset, including attrs of coords, data variables, dimensions, 
    and global attrs, into a single level dictionary for saving to netcdf.
    Returns a copy of the dataset with the changed attributes"""

    def _flatten_dict(d: dict, seperator: str='_._', flat_dict: dict={}, parent_key: str='') -> dict:
        """Flattens a nested dictionary into a single level dictionary."""
        for key, value in d.items():
            
            new_key = f"{parent_key}{seperator}{key}" if parent_key else key
            if isinstance(value, dict):
                flat_dict = _flatten_dict(value, flat_dict=flat_dict, parent_key=new_key)
            else:
                flat_dict[new_key] = value
            
        return flat_dict

    def _flatten_attrs(data: HasAttrs) -> None:
        data.attrs = _flatten_dict(data.attrs, flat_dict={}, parent_key='')

    ds = format_all_attrs(ds, _flatten_attrs)
    return ds


def unflatten_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Unflattens the attributes of the dataset, including attrs of coords, data variables, dimensions,
    and global attrs, into a nested dictionary.
    Returns a copy of the dataset with the changed attributes"""

    def _unflatten_dict(d: dict, seperator: str='_._') -> dict:
        """Unflattens a single level dictionary into a nested dictionary using the seperator to determine nesting."""
        nested_dict = {}
        for key, value in d.items():
            keys = key.split(seperator)
            d2 = nested_dict
            for k in keys[:-1]:
                if k not in d2:
                    d2[k] = {}
                d2 = d2[k]        
            d2[keys[-1]] = value
        return nested_dict
    
    def _unflatten_attrs(data: HasAttrs) -> None:
        data.attrs = _unflatten_dict(data.attrs)

    ds = format_all_attrs(ds, _unflatten_attrs)
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
        if contains_unumpy(ds[key]):
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


def contains_unumpy(da: xr.DataArray) -> bool:
    """Checks whether the data array contains unumpy values"""

    import uncertainties

    return isinstance(da.data.take(0), uncertainties.core.AffineScalarFunc)
