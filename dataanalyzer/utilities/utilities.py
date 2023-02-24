"""Utilities module.

This module contains some utility functions that are used in the
dataanalyzer package.

"""

from typing import Tuple, Union
import numpy as np
import decimal


def convert_array_with_unit(
    array: Union[float, list, tuple, np.ndarray]
) -> Tuple[np.ndarray, str, float]:  # sourcery skip: remove-unnecessary-cast
    """Converts an array to a more readable unit.

    Args:
        array (Union[float, list, tuple, np.ndarray]): The array to be converted.

    Raises:
        TypeError: If the array is not a list, tuple or numpy array.

    Returns:
        Tuple[np.ndarray, str, float]: The converted array, the unit prefix and the conversion factor.
    """
    # Define the unit prefixes
    prefix = "yzafpnµm kMGTPEZY"
    shift = decimal.Decimal("1E24")

    # Check if the array is a list, tuple or numpy array
    if isinstance(array, (float, int)):
        # If not so, raise an error
        raise TypeError("Array must be a list, tuple or numpy array.")

    # Determine the maximum value of the array
    # and convert it to a decimal
    max_value = np.max(np.abs(array))
    deci = (decimal.Decimal(str(max_value)) * shift).normalize()

    # Split the decimal into its mantissa and exponent
    try:
        m, e = deci.to_eng_string().split("E")
    except Exception:
        m, e = deci, 0

    # Calculate the conversion factor
    conversion_factor = float(m) / max_value if max_value != 0 else 1

    # Convert the array
    converted_array = np.array(array) * conversion_factor

    # Determine the unit prefix (if e is 8 the prefix is empty)
    unit_prefix = f"{prefix[int(e) // 3]}" if int(e) != 8 else ""

    # Return the converted array, the unit prefix and the conversion factor
    return converted_array, unit_prefix, conversion_factor


def round_on_error(
    value: Union[float, int], error: Union[float, int], n_digits: int = 1
) -> str:
    """Rounds a value and its error to a given number of significant digits.

    Args:
        value (float, int): The value to be rounded.
        error (float, int): The error of the value.
        n_digits (int, optional): The number of significant digits. Defaults to 1.

    Returns:
        str: The rounded value and error as a string.
    """
    from math import isnan

    # Check if the error is not a number, infinite, or zero
    if isnan(error) or not np.isfinite(error) or error == 0:
        # If so, return the value and error as-is
        return f"{value} ± {error}"

    # Calculate the power of the error
    power = int(np.floor(np.log10(error) - np.log10(0.95)))
    # Calculate the power of the rounded error
    power_round = -power + n_digits - 1

    # Round the error to the power of the rounded error
    error_rounded = round(error, power_round)
    # Round the value to the power of the rounded error
    value_rounded = round(value, power_round)

    # Return the rounded value and error as a string
    return (
        f"{value:.{power_round}f} ± {error:.{power_round}f}"
        if power < 0
        else f"{value_rounded} ± {error_rounded}"
    )


def convert_unit_to_str_or_float(
    f: str, x: Union[float, str, None], y: Union[float, str, None]
):
    if type(f) != str:
        raise ValueError(f"f must be a string, not {type(f)}: {f}")

    if x is None or y is None:
        raise ValueError("x and y must be defined")

    if type(x) != type(y):
        raise TypeError("x and y must be of the same type")

    if isinstance(x, str) and isinstance(y, str):
        f = f.replace("x", x).replace("y", y)
        return f.replace("**", "^").replace("(", "{").replace(")", "}")

    elif isinstance(x, (int, float)):
        try:
            evalue = eval(f)
        except Exception as e:
            return 1

        if isinstance(evalue, (float, int)):
            return evalue
        else:
            raise ValueError("f must evaluate to a number")


def group_by_attr(params: list, attr: str):
    """Sorts a list of Valueclasses into groups by a given attribute."""
    params_sorted = sorted(params, key=lambda x: getattr(x, attr))
    sort_attr = [getattr(param, attr) for param in params_sorted]
    param_groups = []
    for i, param in enumerate(params_sorted):
        if i == 0 or sort_attr[i] != sort_attr[i - 1]:
            param_groups.append([param])
        else:
            param_groups[-1].append(param)
    return param_groups


def split_by_attr(parms: list, attr: str, value=None):
    """Splits a list of Valueclasses into two groups by wether a given attribute equals value."""
    group_with = []
    group_without = []

    for param in parms:
        if getattr(param, attr) == value:
            group_with.append(param)
        else:
            group_without.append(param)
    return group_with, group_without
