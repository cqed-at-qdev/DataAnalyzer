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
    from math import isnan

    if isnan(error) or not np.isfinite(error) or error == 0:
        value_rounded = np.nan if isnan(value) else value
        error_rounded = error
        return f"{value_rounded} ± {error_rounded}"

    power = int(np.floor(np.log10(error) - np.log10(0.95)))
    power_round = -power + n_digits - 1

    error_rounded = round(error, power_round)
    value_rounded = round(value, power_round)

    return (
        f"{value:.{power_round}f} ± {error:.{power_round}f}"
        if power <= 0
        else f"{value_rounded} ± {error_rounded}"
    )
