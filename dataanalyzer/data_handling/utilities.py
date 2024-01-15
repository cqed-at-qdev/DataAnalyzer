import xarray as xr
from typing import Protocol, Callable, Any


####################################################################################################
### handling of non-compatible types in attributes #################################################
####################################################################################################


# Typing class specifying that the object has an attrs attribute of type dict[str, any]
class HasAttrs(Protocol):
    attrs: dict[str, Any]


def format_all_attrs(
    ds: xr.Dataset, formatting_fnc: Callable[[HasAttrs], None]
) -> xr.Dataset:
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


def format_attrs(ds: xr.Dataset, attr_type: type, format_fnc: Callable) -> xr.Dataset:
    """Replaces attrs of type attr_type with a representation savable to netcdf, given by format_fnc.
    Also adds a f"{key}__type" attribute to the attrs dict to allow restoring the original type.
    Intented for simple datatypes. For example, None type cannot be saved to netcdf, but str(None) can.
    Returns a copy of the dataset with the changed attributes"""

    def _format_attrs(data: HasAttrs) -> None:
        type_identifiers = {}
        for attr_key, attr_value in data.attrs.items():
            if type(attr_value) == attr_type:
                data.attrs[attr_key] = format_fnc(attr_value)
                type_identifiers[f"{attr_key}__type"] = str(type(attr_value))
        data.attrs.update(type_identifiers)

    ds = format_all_attrs(ds, _format_attrs)
    return ds


def restore_attrs(ds: xr.Dataset, attr_type: type, restore_fnc: Callable) -> xr.Dataset:
    """Replaces attrs of type attr_type with the result of restore_fnc.
    Used to restore attributes that were replaced by format_attrs."""
    type_string = str(attr_type)

    def _restore_attr(data: HasAttrs) -> None:
        to_be_deleted = []
        for attr_key, attr_value in data.attrs.items():
            if attr_key.endswith("__type") and attr_value == type_string:
                attr_main_key = attr_key[:-6]
                data.attrs[attr_main_key] = restore_fnc(data.attrs[attr_main_key])
                to_be_deleted.append(attr_key)
        for key in to_be_deleted:
            del data.attrs[key]

    ds = format_all_attrs(ds, _restore_attr)

    return ds


def format_none_type(ds: xr.Dataset) -> xr.Dataset:
    return format_attrs(ds, type(None), lambda x: "None")


def restore_none_type(ds: xr.Dataset) -> xr.Dataset:
    return restore_attrs(ds, type(None), lambda x: None)


def format_ufloat_type(ds: xr.Dataset) -> xr.Dataset:
    from uncertainties.core import Variable

    return format_attrs(ds, Variable, lambda x: [x._nominal_value, x.std_dev])


def restore_ufloat_type(ds: xr.Dataset) -> xr.Dataset:
    from uncertainties import ufloat
    from uncertainties.core import Variable

    return restore_attrs(ds, Variable, lambda x: ufloat(*x))


def flatten_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Flattens the attributes of the dataset, including attrs of coords, data variables, dimensions,
    and global attrs, into a single level dictionary for saving to netcdf.
    Returns a copy of the dataset with the changed attributes"""

    def _flatten_dict(
        d: dict, seperator: str = "_._", flat_dict: dict = {}, parent_key: str = ""
    ) -> dict:
        """Flattens a nested dictionary into a single level dictionary."""
        for key, value in d.items():
            new_key = f"{parent_key}{seperator}{key}" if parent_key else key
            if isinstance(value, dict):
                flat_dict = _flatten_dict(
                    value, flat_dict=flat_dict, parent_key=new_key
                )
            else:
                flat_dict[new_key] = value

        return flat_dict

    def _flatten_attrs(data: HasAttrs) -> None:
        data.attrs = _flatten_dict(data.attrs, flat_dict={}, parent_key="")

    ds = format_all_attrs(ds, _flatten_attrs)
    return ds


def unflatten_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Unflattens the attributes of the dataset, including attrs of coords, data variables, dimensions,
    and global attrs, into a nested dictionary.
    Returns a copy of the dataset with the changed attributes"""

    def _unflatten_dict(d: dict, seperator: str = "_._") -> dict:
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


def separate_from_unumpy(ds: xr.Dataset) -> xr.Dataset:
    """Separate data with unumpy errors out to separate dimension to enable dataset saving"""
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
    """Combines data with errors along separate dimension to unumpy array"""
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


####################################################################################################
### complex data handling functions ######################################################################
####################################################################################################


def separate_complex(ds: xr.Dataset) -> xr.Dataset:
    """Separate complex data to separate dimension to enable dataset saving"""

    ds = ds.copy()
    ds.coords["real_imag"] = ["real", "imag"]
    ds.coords["real_imag"].attrs["description"] = "real and imaginary parts"

    for key in ds.data_vars:
        if contains_complex(ds[key]):
            data = ds[key].data
            ds[key] = ds[key].astype("float")
            ds[key] = ds[key].expand_dims({"real_imag": 2}).copy()
            ds[key].loc[{"real_imag": "real"}] = data.real
            ds[key].loc[{"real_imag": "imag"}] = data.imag

    return ds


def combine_to_complex(ds: xr.Dataset) -> xr.Dataset:
    """Combines data with errors along separate dimension to unumpy array"""

    ds = ds.copy()
    if "real_imag" not in ds.coords:
        return ds
    for key in ds.data_vars:
        if "real_imag" in ds[key].dims:
            data_complex = (
                ds[key].loc[{"real_imag": "real"}].data
                + 1j * ds[key].loc[{"real_imag": "imag"}].data
            )

            ds[key] = ds[key].astype(complex)
            ds[key].loc[{"real_imag": "real"}] = data_complex

            # Remove error dimension from data array
            ds[key] = ds[key].sel({"real_imag": "real"}, drop=True)

    ds = ds.drop_dims("real_imag")
    return ds


def contains_complex(da: xr.DataArray) -> bool:
    """Checks whether the data array contains complex values"""

    return isinstance(da.data.take(0), complex)
