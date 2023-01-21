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
    conversion_factor = float(m) / max_value

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
