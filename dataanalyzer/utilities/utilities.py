from typing import Tuple, Union
import numpy as np
import decimal

def convert_array_with_unit(
    array: Union[float, list, tuple, np.ndarray]
) -> Tuple[np.ndarray, str, float]:
    # sourcery skip: remove-unnecessary-cast
    prefix = "yzafpnµm kMGTPEZY"
    shift = decimal.Decimal("1E24")

    if isinstance(array, (float, int)):
        raise TypeError("Array must be a list, tuple or numpy array.")

    max_value = np.max(np.abs(array))

    d = (decimal.Decimal(str(max_value)) * shift).normalize()
    try:
        m, e = d.to_eng_string().split("E")
    except Exception:
        m, e = d, 0

    conversion_factor = float(m) / max_value
    converted_array = np.array(array) * conversion_factor

    unit_prefix = f"{prefix[int(e) // 3]}"

    if unit_prefix == " ":
        unit_prefix = ""

    return converted_array, unit_prefix, conversion_factor


def round_on_error(value, error, n_digits=0):
    if np.isfinite(error):
        if error == 0:
            return value, error

        power = int(np.floor(np.log10(error) - np.log10(0.95)))

        error_rounded = round(error, -power + n_digits)
        value_rounded = round(value, -power + n_digits)
        if power >= 0 and value >= 0:
            error_rounded = int(error_rounded)
            value_rounded = int(value_rounded)
    else:
        from math import isnan

        value_rounded = np.nan if isnan(value) else int(value)
        error_rounded = error
    return value_rounded, error_rounded