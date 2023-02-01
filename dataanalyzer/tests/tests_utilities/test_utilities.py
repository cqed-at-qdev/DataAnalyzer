import contextlib
from dataanalyzer.utilities import utilities
import numpy as np


def _make_function_test(arg0, arg1):
    (
        converted_array,
        unit_prefix,
        conversion_factor,
    ) = utilities.convert_array_with_unit(np.array([1, 2, 3, 4, 5]) * arg0)
    assert all(converted_array == np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert unit_prefix == arg1
    assert conversion_factor == 1 / arg0


def test_convert_array_with_unit():
    _make_function_test(1, " ")


def test_convert_array_with_unit_k():
    _make_function_test(1e3, "k")


def test_convert_array_with_unit_M():
    _make_function_test(1e6, "M")


def test_convert_array_with_unit_G():
    _make_function_test(1e9, "G")


def test_convert_array_with_unit_m():
    _make_function_test(1e-3, "m")


def test_round_on_error():
    assert utilities.round_on_error(1.23456789, 0.00000001) == "1.23456789 ± 0.00000001"
    assert utilities.round_on_error(1.23456789, 0.0000001) == "1.2345679 ± 0.0000001"
    assert utilities.round_on_error(1.23456789, 0.000001) == "1.234568 ± 0.000001"
    assert utilities.round_on_error(1.23456789, 0.00001) == "1.23457 ± 0.00001"
    assert utilities.round_on_error(1.23456789, 0.0001) == "1.2346 ± 0.0001"
    assert utilities.round_on_error(1.23456789, 0.001) == "1.235 ± 0.001"
    assert utilities.round_on_error(1.23456789, 0.01) == "1.23 ± 0.01"
    assert utilities.round_on_error(1.23456789, 0.1) == "1.2 ± 0.1"
    assert utilities.round_on_error(1.23456789, 1) == "1.0 ± 1"
    assert utilities.round_on_error(1.23456789, 10) == "0.0 ± 10"
    assert utilities.round_on_error(1.23456789, 100) == "0.0 ± 100"

    # Test negative error
    with contextlib.suppress(ValueError):
        utilities.round_on_error(1.23456789, -1)
