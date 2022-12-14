# Author: Malthe Asmus Marciniak Nielsen
from typing import Iterable, Optional
from dataclasses import dataclass
import numpy as np

####################################################################################################
#                   Fit parameter dataclass                                                        #
####################################################################################################
@dataclass
class Fitparam:
    values: Optional[float] = 0
    errors: Optional[float] = None
    limits: Optional[Iterable[float]] = (-np.inf, np.inf)
    fixed: bool = False

    def __repr__(self):
        if self.errors is None:
            return f"Fitparam(values={self.values}, limits={self.limits}, fixed={self.fixed})"
        return f"Fitparam(values={self.values}, errors={self.errors}, limits={self.limits}, fixed={self.fixed})"

    def __eq__(self, other):
        if isinstance(other, Fitparam):
            return (
                self.values == other.values
                and self.errors == other.errors
                and self.limits == other.limits
                and self.fixed == other.fixed
            )
        else:
            return False

    def update(self, other):
        if isinstance(other, Fitparam):
            self.values = other.values
            self.limits = other.limits
            self.fixed = other.fixed
        else:
            raise TypeError(f"Can only update with another Fitparam, not {type(other)}")


####################################################################################################
#                   Genereal Functions from Lmfit                                                  #
####################################################################################################
log2 = np.log(2)
s2pi = np.sqrt(2 * np.pi)
s2 = np.sqrt(2.0)
# tiny had been numpy.finfo(numpy.float64).eps ~=2.2e16.
# here, we explicitly set it to 1.e-15 == numpy.finfo(numpy.float64).resolution
tiny = 1.0e-15


def guess_from_peak(y, x, negative, ampscale=1.0, sigscale=1.0):
    """Estimate starting values from 1D peak data and create Parameters."""
    sort_increasing = np.argsort(x)
    x = x[sort_increasing]
    y = y[sort_increasing]

    maxy, miny = max(y), min(y)
    maxx, minx = max(x), min(x)
    cen = x[np.argmax(y)]
    height = (maxy - miny) * 3.0
    sig = (maxx - minx) / 6.0

    # the explicit conversion to a NumPy array is to make sure that the
    # indexing on line 65 also works if the data is supplied as pandas.Series
    x_halfmax = np.array(x[y > (maxy + miny) / 2.0])
    if negative:
        height = -(maxy - miny) * 3.0
        x_halfmax = x[y < (maxy + miny) / 2.0]
    if len(x_halfmax) > 2:
        sig = (x_halfmax[-1] - x_halfmax[0]) / 2.0
        cen = x_halfmax.mean()
    amp = height * sig * ampscale
    sig = sig * sigscale

    amp = Fitparam(values=amp)
    cen = Fitparam(values=cen)
    sig = Fitparam(values=sig)

    return amp, cen, sig


def not_zero(value):
    """Return value with a minimal absolute size of tiny, preserving the sign.

    This is a helper function to prevent ZeroDivisionError's.

    Parameters
    ----------
    value : scalar
        Value to be ensured not to be zero.

    Returns
    -------
    scalar
        Value ensured not to be zero.

    """
    return float(np.copysign(max(tiny, abs(value)), value))


def common_member(*lists):
    return bool(set.intersection(*map(set, lists)))
