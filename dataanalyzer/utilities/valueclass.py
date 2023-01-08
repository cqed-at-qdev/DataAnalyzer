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

    value: Union[float, list, tuple, np.ndarray] = ()  # type: ignore
    error: Union[float, list, tuple, np.ndarray] = ()  # type: ignore
    name: str = ""
    unit: str = ""
    fft_type: Union[str, bool] = False

    def __repr__(self):
        """Returns a string representation of the Valueclass object.

        Returns:
            str : String representation of the Valueclass object.
        """
        error = np.nan if np.isnan(self.error).all() else self.error
        return (
            f"{self.name}:\n(value={self.value}, error={error}, unit={self.unit})\n\n"
        )

    def __getitem__(self, key) -> "Valueclass":
        """Returns a slice of the Valueclass object.

        Args:
            key (str, slice, int): Key to slice the Valueclass object.

        Returns:
            item: Sliced Valueclass object.
        """
        if isinstance(key, (slice, np.integer, int, np.ndarray, list, tuple)):
            return Valueclass(self.value[key], self.error[key], self.name, self.unit)
        return self[key]

    def __setitem__(self, key, value):
        if isinstance(key, (slice, np.integer, int, np.ndarray, list, tuple)):
            if not isinstance(value, Valueclass):
                value = Valueclass(value, self.error[key], self.name, self.unit)
            self.value[key] = value.value
            self.error[key] = value.error
        else:
            setattr(self, key, value)

    @property
    def value(self) -> np.ndarray:
        """Returns the value of the Valueclass object."""
        if not hasattr(self, "_value"):
            self.value = ()

        return self._value

    @value.setter
    def value(self, value: Union[float, list, tuple, np.ndarray]) -> None:
        """Sets the value of the Valueclass object.

        Args:
            value (Union[float, list, tuple, np.ndarray]): Value to set.
        """
        if isinstance(value, (float, int, np.integer)):
            self._value = np.array([value])

        elif isinstance(value, (list, tuple, np.ndarray)):
            self._value = np.array(value)

    @property
    def error(self) -> np.ndarray:
        """Returns the error of the Valueclass object."""
        return self._error

    @error.setter
    def error(self, error: Union[float, list, tuple, np.ndarray]) -> None:
        """Sets the error of the Valueclass object.

        Args:
            error (Union[float, list, tuple, np.ndarray]): Error to set.
        """
        self._error = (
            np.full(np.shape(self.value), np.nan) if self.value.size else np.empty(0)
        )
        if isinstance(error, (float, np.integer)):
            self._error.fill(error)

        elif isinstance(error, (list, tuple, np.ndarray)):
            error = np.array(error)
            ndim = np.ndim(error)
            if ndim == 0:
                self._error.fill(error)
            elif ndim == 1:
                self._error[: np.size(error)] = error
            elif ndim == 2:
                w, h = np.shape(error)[0], np.shape(error)[1]

                if w == 1 or h == 1:
                    self._error = error
                else:
                    self._error[:w, :h] = error

    @property
    def v(self):
        """Returns the value of the Valueclass object."""
        return self.value

    @property
    def e(self):
        """Returns the error of the Valueclass object."""
        return self.error

    def __add__(self, other) -> "Valueclass":
        """Adds two Valueclass objects or a Valueclass object and a float.

        Args:
            other (object, float, int): Object to add.

        Returns:
            Valueclass: Sum of the two objects.
        """
        if "Valueclass" not in str(type(other)):
            return Valueclass(
                self.value + other, self.error, self.name, self.unit, self.fft_type,
            )
        else:
            return Valueclass(
                self.value + other.value,
                np.sqrt(self.error ** 2 + other.error ** 2),
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
                self.value - other, self.error, self.name, self.unit, self.fft_type,
            )
        else:
            return Valueclass(
                self.value - other.value,
                np.sqrt(self.error ** 2 + other.error ** 2),
                self.name,
                self.unit,
            )

    def __mul__(self, other) -> "Valueclass":
        if "Valueclass" not in str(type(other)):
            return Valueclass(
                self.value * other,
                self.error * other,
                self.name,
                self.unit,
                self.fft_type,
            )
        else:
            return Valueclass(
                self.value * other.value,
                np.sqrt(
                    (self.error * other.value) ** 2 + (self.value * other.error) ** 2
                ),
                self.name,
                self.unit,
            )

    def __truediv__(self, other) -> "Valueclass":
        if "Valueclass" not in str(type(other)):
            return Valueclass(
                self.value / other,
                self.error / other,
                self.name,
                self.unit,
                self.fft_type,
            )
        else:
            return Valueclass(
                self.value / other.value,
                np.sqrt(
                    (self.error / other.value) ** 2
                    + (self.value * other.error / other.value ** 2) ** 2
                ),
                self.name,
                self.unit,
            )

    def __pow__(self, other) -> "Valueclass":
        if "Valueclass" not in str(type(other)):
            return Valueclass(
                self.value ** other,
                self.error * other * self.value ** (other - 1),
                self.name,
                self.unit,
            )
        else:
            return Valueclass(
                self.value ** other.value,
                np.sqrt(
                    (self.error * other.value * self.value ** (other.value - 1)) ** 2
                    + (self.value ** other.value * other.error * np.log(self.value))
                    ** 2
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
            other / self.value,
            other * self.error / self.value ** 2,
            self.name,
            self.unit,
            self.fft_type,
        )

    def __rpow__(self, other) -> "Valueclass":
        return Valueclass(
            other ** self.value,
            other ** self.value * self.error * np.log(other),
            self.name,
            self.unit,
            self.fft_type,
        )

    def __len__(self):
        return len(self.value)

    def __max__(self):
        return max(self.value)

    @property
    def db(self):
        return Valueclass(
            20 * np.log10(self.value),
            20 * self.error / (np.log(10) * self.value),
            self.name,
            "dB",
        )

    @property
    def norm(self):
        return Valueclass(
            self.value / np.sqrt(np.sum(self.value ** 2)),
            self.error / np.sqrt(np.sum(self.value ** 2)),
            self.name,
            self.unit,
            self.fft_type,
        )

    @property
    def fft(self):
        fft = np.fft.fft(self.value)
        N = np.size(self.value)
        v = 2.0 / N * np.abs(fft[: N // 2])

        return Valueclass(v, np.nan, self.name, self.unit, fft_type="fft_y")

    @property
    def fftfreq(self):
        N = np.size(self.value)
        return Valueclass(
            np.fft.fftfreq(N, d=self.value[1] - self.value[0])[: N // 2],
            np.nan,
            self.name,
            self.unit,
            fft_type="fft_x",
        )

    @property
    def substract_mean(self):
        return Valueclass(
            self.value - np.mean(self.value), self.error, self.name, self.unit
        )

    def mean(self, axis=None):
        return Valueclass(
            np.mean(self.value, axis=axis),
            np.mean(self.error, axis=axis),
            self.name,
            self.unit,
        )

    def std(self, axis=None):
        return Valueclass(
            np.std(self.value, axis=axis),
            np.std(self.error, axis=axis),
            self.name,
            self.unit,
        )

    @property
    def ddx(self):
        return Valueclass(
            np.gradient(self.value, self.error),
            np.gradient(self.error, self.error),
            self.name,
            self.unit,
            self.fft_type,
        )

    @property
    def ddxx(self):
        return Valueclass(
            np.gradient(np.gradient(self.value, self.error), self.error),
            np.gradient(np.gradient(self.error, self.error), self.error),
            self.name,
            self.unit,
            self.fft_type,
        )

    def norm_zero_to_one(self, axis=None):
        return Valueclass(
            (self.value - np.min(self.value, axis=axis))
            / (np.max(self.value, axis=axis) - np.min(self.value, axis=axis)),
            self.error
            / (np.max(self.value, axis=axis) - np.min(self.value, axis=axis)),
            self.name,
            self.unit,
            self.fft_type,
        )

    def min(self, axis=None):
        return np.min(self.value, axis=axis)

    def max(self, axis=None):
        return np.max(self.value, axis=axis)

    def argmin(self, axis=None):
        return np.argmin(self.value, axis=axis)

    def argmax(self, axis=None):
        return np.argmax(self.value, axis=axis)

    def min_error(self, axis=None):
        return np.min(self.error, axis=axis)

    def max_error(self, axis=None):
        return np.max(self.error, axis=axis)

    def argmin_error(self, axis=None):
        return np.argmin(self.error, axis=axis)

    def argmax_error(self, axis=None):
        return np.argmax(self.error, axis=axis)

    @property
    def real(self):
        return Valueclass(
            np.real(self.value),
            np.real(self.error),
            self.name,
            self.unit,
            self.fft_type,
        )

    @property
    def imag(self):
        return Valueclass(
            np.imag(self.value),
            np.imag(self.error),
            self.name,
            self.unit,
            self.fft_type,
        )

    @property
    def abs(self):
        return Valueclass(np.abs(self.value), np.abs(self.error), self.name, self.unit)

    @property
    def phase(self):
        error = np.unwrap(np.angle(self.error))
        if not np.any(np.isnan(error)):
            print(error)
            signal.detrend(error)

        return Valueclass(
            signal.detrend(np.unwrap(np.angle(self.value))),
            error,
            self.name,
            self.unit,
            self.fft_type,
        )

    @property
    def angle(self):
        return Valueclass(
            np.unwrap(np.angle(self.value)),
            np.unwrap(np.angle(self.error)),
            self.name,
            "rad",
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
                self.value, self.error, self.name, self.unit, self.fft_type,
            )

        elif operation in ("Substract first", "substract first", "first", "First"):
            return Valueclass(
                self.value - self.value[0],
                self.error,
                self.name,
                self.unit,
                self.fft_type,
            )

        elif operation in ("Substract mean", "substract mean", "mean", "Mean"):
            return Valueclass(
                self.value - np.mean(self.value, axis=0),
                self.error,
                self.name,
                self.unit,
                self.fft_type,
            )

        elif operation in ("Substract last", "substract last", "last", "Last"):
            return Valueclass(
                self.value - self.value[-1],
                self.error,
                self.name,
                self.unit,
                self.fft_type,
            )

        elif operation in ("Substract min", "substract min", "min", "Min"):
            return Valueclass(
                self.value - np.min(self.value, axis=0),
                self.error,
                self.name,
                self.unit,
                self.fft_type,
            )

        elif operation in ("Substract max", "substract max", "max", "Max"):
            return Valueclass(
                self.value - np.max(self.value, axis=0),
                self.error,
                self.name,
                self.unit,
                self.fft_type,
            )

        elif operation in ("Substract median", "substract median", "median", "Median"):
            return Valueclass(
                self.value - np.median(self.value, axis=0),
                self.error,
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
            v = self.value - np.roll(self.value, 1, axis=0)  # type: ignore
            v[0] = np.zeros(self.value.shape[1])

            return Valueclass(v, self.error, self.name, self.unit, self.fft_type,)

        elif operation in ("Average", "average"):
            return Valueclass(
                np.tile(np.mean(self.value, axis=0), (np.shape(self.value)[0], 1)),
                np.tile(np.mean(self.error, axis=0), (np.shape(self.value)[0], 1)),
                self.name,
                self.unit,
            )

        elif operation in ("Standard deviation", "standard deviation", "std", "Std"):
            return Valueclass(
                np.tile(np.std(self.value, axis=0), (np.shape(self.value)[0], 1)),
                np.tile(np.std(self.error, axis=0), (np.shape(self.value)[0], 1)),
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
        x = kwargs.pop("x", np.arange(len(self.value)))
        x_label = kwargs.pop("x_label", "index")
        y_label = kwargs.pop("y_label", self.name)
        title = kwargs.pop("title", None)
        fmt = kwargs.pop("fmt", ".")

        plt.errorbar(x, self.value, yerr=self.error, fmt=fmt, *args, **kwargs)
        plt.plot(x, self.value, *args, **kwargs)
        plt.xlabel(x_label)
        plt.ylabel(f"{y_label} [{self.unit}]")

        if title is not None:
            plt.title(title)

        plt.show()

    def asdict(self):
        self_copy = copy.copy(self)
        self_copy.value = self_copy.value.tolist()
        self_copy.error = self_copy.error.tolist()
        return asdict(self_copy)

    @staticmethod
    def fromdict(newdict: dict):
        if isinstance(newdict["value"], dict):
            newdict["value"] = newdict["value"]["I"] + 1j * newdict["value"]["Q"]

        if isinstance(newdict["error"], dict):
            newdict["error"] = newdict["error"]["I"] + 1j * newdict["error"]["Q"]

        return Valueclass(**newdict)

    @property
    def shape(self):
        return self.value.shape

    @property
    def size(self):
        return self.value.size

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def T(self):
        return Valueclass(
            self.value.T, self.error.T, self.name, self.unit, self.fft_type
        )

    # def clip(self, a_min=None, a_max=None, out=None):
    #     # TODO: Check if this is correct
    #     return Valueclass(
    #         np.clip(self.v, a_min, a_max, out),
    #         np.clip(self.e, a_min, a_max, out),
    #         self.name,
    #         self.unit,
    #         self.fft_type,
    #     )

    @property
    def sprt(self):
        return Valueclass(
            np.sqrt(self.value),
            self.error / (2 * np.sqrt(self.value)),
            self.name,
            self.unit,
            self.fft_type,
        )

    def append(self, other, axis=None):
        if not isinstance(other, Valueclass):
            other = from_float_to_valueclass(other, self.name)

        self.value = np.append(self.value, other.value, axis=axis)
        self.error = np.append(self.error, other.error, axis=axis)


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

    #################    Example 2    #################
    # make an empty Valueclass object
    test = Valueclass(name="y", unit="V")

    #################    Example 3    #################
    x = np.linspace(0, 10, 50)
    y = np.sin(x) + np.random.normal(0, 0.1, 50)

    # make Valueclass objects
    test1 = Valueclass(y, name="y", unit="V")
    test2 = Valueclass(x, name="x", unit="V")

    test1 += test2
    test1[:10] = Valueclass(x, name="x", unit="V")[:10]
    test1.plot()

