# Author: Malthe Asmus Marciniak Nielsen
from typing import Iterable, Optional
from dataclasses import dataclass
import numpy as np

####################################################################################################
#                   Fit parameter dataclass                                                        #
####################################################################################################
@dataclass
class Fitparam:
    base_name: str = ""
    values: Optional[float] = 0
    errors: Optional[float] = None
    limits: Optional[Iterable[float]] = (-np.inf, np.inf)
    fixed: bool = False

    model: "ModelABC" = None  # type: ignore

    def __repr__(self):
        repr_str = "Fitparam("

        if self.base_name:
            repr_str += f"base_name={self.base_name}"
        if self.full_name != self.base_name:
            repr_str += f", full_name={self.full_name}"
        if self.display_name:
            repr_str += f", display_name={self.display_name}"
        if self.unit is not None:
            repr_str += f", unit={self.unit}"
        if self.values is not None:
            repr_str += f", values={self.values}"
        if self.errors is not None:
            repr_str += f", errors={self.errors}"
        if self.limits is not None:
            repr_str += f", limits={self.limits}"
        if self.fixed is not None:
            repr_str += f", fixed={self.fixed}"
        repr_str += ")"

        return repr_str

    @property
    def full_name(self):
        self.full_name = None

        return self._full_name

    @full_name.setter
    def full_name(self, value):
        if value is None:
            if self.model:
                self._full_name = (
                    self.model._prefix + self.base_name + self.model._suffix
                )
            else:
                self._full_name = self.base_name
        else:
            self._full_name = value

    @property
    def display_name(self):
        if not hasattr(self, "_display_name"):
            self.display_name = None
            self._costom_display_name = False

        if not self._costom_display_name:
            self.display_name = None

        return self._display_name

    @display_name.setter
    def display_name(self, value):
        if value is None:
            if self.model:
                self._display_name = (
                    self.model._prefix + self.symbol + self.model._suffix
                )
            else:
                self._display_name = self.symbol
        else:
            self._costom_display_name = True
            self._display_name = value

    @property
    def unit(self) -> str:
        if not hasattr(self, "_unit"):
            self.unit = None
        return self._unit

    @unit.setter
    def unit(self, value):
        if value is None:
            self._unit = (
                self.model.units[self.base_name]
                if hasattr(self.model, "units") and self.base_name in self.model.units
                else ""
            )
        else:
            self._unit = value

    @property
    def symbol(self):
        if not hasattr(self, "_symbol"):
            self.symbol = None

        return self._symbol

    @symbol.setter
    def symbol(self, value):
        if value is None:
            self._symbol = (
                self.model.symbols[self.base_name]
                if hasattr(self.model, "symbols")
                and self.base_name in self.model.symbols
                else self.base_name
            )
        else:
            self._symbol = value

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
    sig = Fitparam(values=sig, limits=(0, np.inf))

    return amp, cen, sig


def guess_from_multipeaks(y, x, negative, ampscale=1.0, sigscale=1.0, n_peaks=1):
    from scipy.signal import find_peaks

    sort_increasing = np.argsort(x)
    x = x[sort_increasing]
    y = y[sort_increasing]
    deltax = x[1] - x[0]

    if negative:
        y = -y

    # Find peaks

    peaks, properties = find_peaks(y, prominence=0, width=0)

    cens = x[peaks]
    amps = properties["prominences"]
    sigmas = properties["widths"]

    # Sort peaks
    sort_increasing = np.argsort(amps)
    cens = cens[sort_increasing]
    amps = amps[sort_increasing] * ampscale
    sigmas = sigmas[sort_increasing] * deltax * sigscale

    # Convert to Fitparam
    guess = {}
    for i in range(len(cens)):
        index = "" if n_peaks == 1 else f"_{i + 1}"
        guess[f"amplitude{index}"] = Fitparam(values=amps[i])
        guess[f"center{index}"] = Fitparam(values=cens[i])
        guess[f"sigma{index}"] = Fitparam(values=sigmas[i], limits=(0, np.inf))
    return guess


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
