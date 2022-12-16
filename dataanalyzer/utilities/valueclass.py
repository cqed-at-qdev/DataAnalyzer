from dataclasses import dataclass, asdict
from typing import Union
import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


@dataclass
class Valueclass:
    """Valueclass class for storing values and errors. The class is designed to be used in a similar way as the numpy.ndarray class. It can be sliced, added, subtracted, multiplied and divided with other Valueclass objects or with floats.

    Returns:
        _type_: _description_
    """

    value: Union[float, list, tuple, np.ndarray]  # type: ignore
    error: Union[float, list, tuple, np.ndarray] = ()  # type: ignore
    name: str = ""
    unit: str = ""
    fft_type: Union[str, bool] = False

    def __repr__(self):
        """Returns a string representation of the Valueclass object.

        Returns:
            str : String representation of the Valueclass object.
        """
        error = np.nan if np.isnan(self.e).all() else self.e
        return f"{self.name}:\n(value={self.v}, error={error}, unit={self.unit})\n\n"

    def __getitem__(self, key) -> "Valueclass":
        """Returns a slice of the Valueclass object.

        Args:
            key (str, slice, int): Key to slice the Valueclass object.

        Returns:
            item: Sliced Valueclass object.
        """
        if isinstance(key, (slice, np.integer, int, np.ndarray, list, tuple)):
            return Valueclass(self.v[key], self.e[key], self.name, self.unit)
        return self[key]

    @property
    def value(self) -> np.ndarray:
        """Returns the value of the Valueclass object."""
        return self.v

    @value.setter
    def value(self, value: Union[float, list, tuple, np.ndarray]) -> None:
        """Sets the value of the Valueclass object.

        Args:
            value (Union[float, list, tuple, np.ndarray]): Value to set.
        """
        if isinstance(value, (float, np.integer)):
            self.v = np.array([value])
        elif isinstance(value, (list, tuple, np.ndarray)):
            self.v = np.array(value)

    @property
    def error(self) -> np.ndarray:
        """Returns the error of the Valueclass object."""
        return self.e

    @error.setter
    def error(self, error: Union[float, list, tuple, np.ndarray]) -> None:
        """Sets the error of the Valueclass object.

        Args:
            error (Union[float, list, tuple, np.ndarray]): Error to set.
        """
        self.e: np.ndarray = np.full(np.shape(self.v), np.nan)

        if isinstance(error, (float, np.integer)):
            self.e.fill(error)

        elif isinstance(error, (list, tuple, np.ndarray)):
            error = np.array(error)
            ndim = np.ndim(error)
            if ndim == 0:
                self.e.fill(error)
            elif ndim == 1:
                self.e[: np.size(error)] = error
            elif ndim == 2:
                w, h = np.shape(error)[0], np.shape(error)[1]

                if w == 1 or h == 1:
                    self.e = error
                else:
                    self.e[:w, :h] = error

    def __add__(self, other) -> "Valueclass":
        """Adds two Valueclass objects or a Valueclass object and a float.

        Args:
            other (object, float, int): Object to add.

        Returns:
            Valueclass: Sum of the two objects.
        """
        if "Valueclass" not in str(type(other)):
            return Valueclass(
                self.v + other,
                self.e,
                self.name,
                self.unit,
                self.fft_type,
            )
        else:
            return Valueclass(
                self.v + other.v,
                np.sqrt(self.e**2 + other.e**2),
                self.name,
                self.unit,
            )

    def __sub__(self, other) -> "Valueclass":
        """Subtracts two Valueclass objects or a Valueclass object and a float.

        Args:
            other (object, float, int): Object to subtract.

        Returns:
            Valueclass: Difference of the two objects.
        """
        if "Valueclass" not in str(type(other)):
            return Valueclass(
                self.v - other,
                self.e,
                self.name,
                self.unit,
                self.fft_type,
            )
        else:
            return Valueclass(
                self.v - other.v,
                np.sqrt(self.e**2 + other.e**2),
                self.name,
                self.unit,
            )

    def __mul__(self, other) -> "Valueclass":
        if "Valueclass" not in str(type(other)):
            return Valueclass(
                self.v * other,
                self.e * other,
                self.name,
                self.unit,
                self.fft_type,
            )
        else:
            return Valueclass(
                self.v * other.v,
                np.sqrt((self.e * other.v) ** 2 + (self.v * other.e) ** 2),
                self.name,
                self.unit,
            )

    def __truediv__(self, other) -> "Valueclass":
        if "Valueclass" not in str(type(other)):
            return Valueclass(
                self.v / other,
                self.e / other,
                self.name,
                self.unit,
                self.fft_type,
            )
        else:
            return Valueclass(
                self.v / other.v,
                np.sqrt(
                    (self.e / other.v) ** 2 + (self.v * other.e / other.v**2) ** 2
                ),
                self.name,
                self.unit,
            )

    def __pow__(self, other) -> "Valueclass":
        if "Valueclass" not in str(type(other)):
            return Valueclass(
                self.v**other,
                self.e * other * self.v ** (other - 1),
                self.name,
                self.unit,
            )
        else:
            return Valueclass(
                self.v**other.v,
                np.sqrt(
                    (self.e * other.v * self.v ** (other.v - 1)) ** 2
                    + (self.v**other.v * other.e * np.log(self.v)) ** 2
                ),
                self.name,
                self.unit,
            )

    def __radd__(self, other) -> "Valueclass":
        return self + other

    def __rsub__(self, other) -> "Valueclass":
        return other - self

    def __rmul__(self, other) -> "Valueclass":
        return self * other

    def __rtruediv__(self, other) -> "Valueclass":
        return Valueclass(
            other / self.v,
            other * self.e / self.v**2,
            self.name,
            self.unit,
            self.fft_type,
        )

    def __rpow__(self, other) -> "Valueclass":
        return Valueclass(
            other**self.v,
            other**self.v * self.e * np.log(other),
            self.name,
            self.unit,
            self.fft_type,
        )

    @property
    def db(self):
        return Valueclass(
            20 * np.log10(self.v),
            20 * self.e / (np.log(10) * self.v),
            self.name,
            "dB",
        )

    @property
    def norm(self):
        return Valueclass(
            self.v / np.sqrt(np.sum(self.v**2)),
            self.e / np.sqrt(np.sum(self.v**2)),
            self.name,
            self.unit,
            self.fft_type,
        )

    @property
    def fft(self):
        fft = np.fft.fft(self.v)
        N = np.size(self.v)
        v = 2.0 / N * np.abs(fft[: N // 2])

        return Valueclass(v, np.nan, self.name, self.unit, fft_type="fft_y")

    @property
    def fftfreq(self):
        N = np.size(self.v)
        return Valueclass(
            np.fft.fftfreq(N, d=self.v[1] - self.v[0])[: N // 2],
            np.nan,
            self.name,
            self.unit,
            fft_type="fft_x",
        )

    @property
    def substract_mean(self):
        return Valueclass(self.v - np.mean(self.v), self.e, self.name, self.unit)

    def mean(self, axis=None):
        return Valueclass(
            np.mean(self.v, axis=axis), np.mean(self.e, axis=axis), self.name, self.unit
        )

    def std(self, axis=None):
        return Valueclass(
            np.std(self.v, axis=axis), np.std(self.e, axis=axis), self.name, self.unit
        )

    @property
    def ddx(self):
        return Valueclass(
            np.gradient(self.v, self.e),
            np.gradient(self.e, self.e),
            self.name,
            self.unit,
            self.fft_type,
        )

    @property
    def ddxx(self):
        return Valueclass(
            np.gradient(np.gradient(self.v, self.e), self.e),
            np.gradient(np.gradient(self.e, self.e), self.e),
            self.name,
            self.unit,
            self.fft_type,
        )

    def norm_zero_to_one(self, axis=None):
        return Valueclass(
            (self.v - np.min(self.v, axis=axis))
            / (np.max(self.v, axis=axis) - np.min(self.v, axis=axis)),
            self.e / (np.max(self.v, axis=axis) - np.min(self.v, axis=axis)),
            self.name,
            self.unit,
            self.fft_type,
        )

    def min(self, axis=None):
        return np.min(self.v, axis=axis)

    def max(self, axis=None):
        return np.max(self.v, axis=axis)

    def argmin(self, axis=None):
        return np.argmin(self.v, axis=axis)

    def argmax(self, axis=None):
        return np.argmax(self.v, axis=axis)

    def min_error(self, axis=None):
        return np.min(self.e, axis=axis)

    def max_error(self, axis=None):
        return np.max(self.e, axis=axis)

    def argmin_error(self, axis=None):
        return np.argmin(self.e, axis=axis)

    def argmax_error(self, axis=None):
        return np.argmax(self.e, axis=axis)

    @property
    def real(self):
        return Valueclass(
            np.real(self.v),
            np.real(self.e),
            self.name,
            self.unit,
            self.fft_type,
        )

    @property
    def imag(self):
        return Valueclass(
            np.imag(self.v),
            np.imag(self.e),
            self.name,
            self.unit,
            self.fft_type,
        )

    @property
    def abs(self):
        return Valueclass(np.abs(self.v), np.abs(self.e), self.name, self.unit)

    @property
    def phase(self):
        error = np.unwrap(np.angle(self.e))
        if not np.any(np.isnan(error)):
            print(error)
            signal.detrend(error)

        return Valueclass(
            signal.detrend(np.unwrap(np.angle(self.v))),
            error,
            self.name,
            self.unit,
            self.fft_type,
        )

    @property
    def angle(self):
        return Valueclass(
            np.unwrap(np.angle(self.v)),
            np.unwrap(np.angle(self.e)),
            self.name,
            self.unit,
            self.fft_type,
        )

    def traces(self, operation="Show individual"):
        if operation in (
            "Show individual",
            "show individual",
            "individual",
            "Individual",
        ):
            return Valueclass(
                self.v,
                self.e,
                self.name,
                self.unit,
                self.fft_type,
            )

        elif operation in ("Substract first", "substract first", "first", "First"):
            return Valueclass(
                self.v - self.v[0],
                self.e,
                self.name,
                self.unit,
                self.fft_type,
            )

        elif operation in ("Substract mean", "substract mean", "mean", "Mean"):
            return Valueclass(
                self.v - np.mean(self.v, axis=0),
                self.e,
                self.name,
                self.unit,
                self.fft_type,
            )

        elif operation in ("Substract last", "substract last", "last", "Last"):
            return Valueclass(
                self.v - self.v[-1],
                self.e,
                self.name,
                self.unit,
                self.fft_type,
            )

        elif operation in ("Substract min", "substract min", "min", "Min"):
            return Valueclass(
                self.v - np.min(self.v, axis=0),
                self.e,
                self.name,
                self.unit,
                self.fft_type,
            )

        elif operation in ("Substract max", "substract max", "max", "Max"):
            return Valueclass(
                self.v - np.max(self.v, axis=0),
                self.e,
                self.name,
                self.unit,
                self.fft_type,
            )

        elif operation in ("Substract median", "substract median", "median", "Median"):
            return Valueclass(
                self.v - np.median(self.v, axis=0),
                self.e,
                self.name,
                self.unit,
                self.fft_type,
            )

        elif operation in (
            "Substract previous",
            "substract previous",
            "previous",
            "Previous",
        ):
            v = self.v - np.roll(self.v, 1, axis=0)  # type: ignore
            v[0] = np.zeros(self.v.shape[1])

            return Valueclass(
                v,
                self.e,
                self.name,
                self.unit,
                self.fft_type,
            )

        elif operation in ("Average", "average"):
            return Valueclass(
                np.tile(np.mean(self.v, axis=0), (np.shape(self.v)[0], 1)),
                np.tile(np.mean(self.e, axis=0), (np.shape(self.v)[0], 1)),
                self.name,
                self.unit,
            )

        elif operation in ("Standard deviation", "standard deviation", "std", "Std"):
            return Valueclass(
                np.tile(np.std(self.v, axis=0), (np.shape(self.v)[0], 1)),
                np.tile(np.std(self.e, axis=0), (np.shape(self.v)[0], 1)),
                self.name,
                self.unit,
            )

        else:
            raise ValueError(
                f"Operation '{operation}' not recognized\n"
                "self.traces() takes one of the following arguments:"
                "'Show individual', 'Substract first', 'Substract mean',"
                "'Substract last', 'Substract min', 'Substract max', 'Substract median',"
                "'Substract previous', 'Average', 'Standard deviation'"
            )

    def plot(self, *args, **kwargs):
        x = kwargs.pop("x", np.arange(len(self.v)))
        x_label = kwargs.pop("x_label", "index")
        y_label = kwargs.pop("y_label", self.name)
        title = kwargs.pop("title", None)
        fmt = kwargs.pop("fmt", ".")

        plt.errorbar(x, self.v, yerr=self.e, fmt=fmt, *args, **kwargs)
        plt.plot(x, self.v, *args, **kwargs)
        plt.xlabel(x_label)
        plt.ylabel(f"{y_label} [{self.unit}]")

        if title is not None:
            plt.title(title)

        plt.show()

    def asdict(self):
        self_copy = copy.copy(self)
        self_copy.v = self_copy.v.tolist()
        self_copy.e = self_copy.e.tolist()
        return asdict(self_copy)

    @staticmethod
    def fromdict(dict):
        return Valueclass(**dict)

    @property
    def shape(self):
        return self.v.shape

    @property
    def size(self):
        return self.v.size

    @property
    def ndim(self):
        return self.v.ndim

    @property
    def dtype(self):
        return self.v.dtype

    @property
    def T(self):
        return Valueclass(self.v.T, self.e.T, self.name, self.unit, self.fft_type)

    def clip(self, a_min=None, a_max=None, out=None):
        return Valueclass(
            np.clip(self.v, a_min, a_max, out),
            np.clip(self.e, a_min, a_max, out),
            self.name,
            self.unit,
            self.fft_type,
        )

    @property
    def sprt(self):
        return Valueclass(
            np.sqrt(self.v),
            self.e / (2 * np.sqrt(self.v)),
            self.name,
            self.unit,
            self.fft_type,
        )


def from_float_to_valueclass(
    value: Union[Valueclass, list, tuple, np.ndarray], name: str
) -> Valueclass:
    """Converts a float to a Valueclass object.

    Args:
        value (float): The value to be converted.
        name (str): The name of the value.

    Returns:
        Valueclass: The converted value.
    """

    return (
        value if isinstance(value, Valueclass) else Valueclass(name=name, value=value)
    )


if __name__ == "__main__":
    #################    Example 1    #################
    # make fake data with error
    x = np.linspace(0, 10, 50)
    y = np.sin(x) + np.random.normal(0, 0.1, 50)
    yerr = np.random.normal(0.1, 0.01, 50)

    # make Valueclass objects
    test = Valueclass(y, yerr, name="y", unit="V")
    test.plot()
