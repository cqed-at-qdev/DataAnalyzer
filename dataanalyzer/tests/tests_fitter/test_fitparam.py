import dataanalyzer.fitter.fitter_classsetup as fitter_classsetup
import numpy as np


def test_Fitparam():
    # create test data
    values = 1
    errors = 2
    limits = (0, 10)
    fixed = True
    # test function 0
    fitparam = fitter_classsetup.Fitparam(
        values=values, errors=errors, limits=limits, fixed=fixed
    )
    # check result 0
    assert fitparam.values == values
    assert fitparam.errors == errors
    assert fitparam.limits == limits
    assert fitparam.fixed == fixed
    # test function 1
    fitparam = fitter_classsetup.Fitparam(
        values=values, errors=None, limits=(-np.inf, np.inf), fixed=True
    )
    # check result 1
    assert fitparam.values == values
    assert fitparam.errors is None
    assert fitparam.limits == (-np.inf, np.inf)
    assert fitparam.fixed is True


def test_guess_from_peak():
    # create test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    # test function
    amp, cen, sig = fitter_classsetup.guess_from_peak(y, x, negative=False)
    # check result
    assert amp.values == 27.86592040331309
    assert cen.values == 4.747474747474746
    assert sig.values == 4.646464646464647
    assert not any([amp.errors, cen.errors, sig.errors])
    assert amp.limits == (-np.inf, np.inf)


def test_guess_from_peak_negative():
    # create test data
    x = np.linspace(0, 10, 100)
    y = -np.sin(x)
    # test function
    amp, cen, sig = fitter_classsetup.guess_from_peak(y, x, negative=True)
    # check result
    assert amp.values == -27.86592040331309
    assert cen.values == 4.747474747474746
    assert sig.values == 4.646464646464647
    assert not any([amp.errors, cen.errors, sig.errors])
    assert amp.limits == (-np.inf, np.inf)


def test_guess_from_multipeaks():
    # create test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.sin(x + 1)
    # test function
    result = fitter_classsetup.guess_from_multipeaks(y, x, negative=False, n_peaks=2)
    amp1, cen1, sig1, amp2, cen2, sig2 = result.values()
    # check result
    assert type(result) == dict
    assert len(result) == 6
    assert amp1.base_name == ""
    assert not any(
        [amp1.errors, cen1.errors, sig1.errors, amp2.errors, cen2.errors, sig2.errors]
    )
    assert amp1.limits == (-np.inf, np.inf)
    assert amp2.limits == (-np.inf, np.inf)


def test_not_zero():
    # test function
    result = fitter_classsetup.not_zero(0)
    # check result
    assert result == 1e-15


def test_common_member():
    # create test data
    a = [1, 2, 3, 4, 5]
    b = [5, 6, 7, 8, 9]
    c = [10, 15, 20, 25]

    # check result
    assert fitter_classsetup.common_member(a, b)
    assert not fitter_classsetup.common_member(a, c)
