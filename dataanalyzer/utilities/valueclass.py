"""Valueclass class for storing values and errors.

The class is designed to be used in a similar way as the numpy.ndarray class.
It can be sliced, added, subtracted, multiplied and divided with other Valueclass objects or with floats.
It uses some of the same functionality as seen in the Labber GUI (See .traces() method).
"""

from dataclasses import asdict, dataclass
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from dataanalyzer.utilities.utilities import convert_array_with_unit, round_on_error


@dataclass
class Valueclass:
    """Valueclass class for storing values and errors.
    The class is designed to be used in a similar way as the numpy.ndarray class.
    It can be sliced, added, subtracted, multiplied and divided with other Valueclass objects or with floats.

    Returns:
        Valueclass: Valueclass object.
    """

    value: Union[float, list, tuple, np.ndarray] = ()  # type: ignore
    error: Union[float, list, tuple, np.ndarray] = ()  # type: ignore
    name: str = ""
    unit: str = ""
    fft_type: Union[str, bool] = False
    sweep_idx: Optional[int] = None

    ####################################################################################################
    #                   Dunder Functions                                                               #
    ####################################################################################################
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

    def __setitem__(self, key, value) -> None:
        """Sets a slice of the Valueclass object.

        Args:
            key (str, slice, int): Key to slice the Valueclass object.
            value (object): Value to set the slice to.
        """
        if isinstance(key, (slice, np.integer, int, np.ndarray, list, tuple)):
            if not isinstance(value, Valueclass):
                value = Valueclass(value, self.error[key], self.name, self.unit)
            self.value[key] = value.value
            self.error[key] = value.error
        else:
            setattr(self, key, value)

    def __add__(self, other) -> "Valueclass":
        """Adds two Valueclass objects or a Valueclass object and a float.

        Args:
            other (object, float, int): Object to add.

        Returns:
            Valueclass: Sum of the two objects.
        """
        if "Valueclass" not in str(type(other)):
            return Valueclass(
                self.value + other,
                self.error,
                self.name,
                self.unit,
                self.fft_type,
            )
        else:
            return Valueclass(
                self.value + other.value,
                np.sqrt(self.error**2 + other.error**2),
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
                self.value - other,
                self.error,
                self.name,
                self.unit,
                self.fft_type,
            )
        else:
            return Valueclass(
                self.value - other.value,
                np.sqrt(self.error**2 + other.error**2),
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
                    + (self.value * other.error / other.value**2) ** 2
                ),
                self.name,
                self.unit,
            )

    def __pow__(self, other) -> "Valueclass":
        if "Valueclass" not in str(type(other)):
            return Valueclass(
                self.value**other,
                self.error
                * other
                * self.value ** (other - 1),  # TODO: make correct error propagation
                self.name,
                self.unit,
            )
        else:
            return Valueclass(
                self.value**other.value,
                np.sqrt(
                    (self.error * other.value * self.value ** (other.value - 1)) ** 2
                    + (self.value**other.value * other.error * np.log(self.value))
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
            other * self.error / self.value**2,
            self.name,
            self.unit,
            self.fft_type,
        )

    def __rpow__(self, other) -> "Valueclass":
        return Valueclass(
            other**self.value,
            other**self.value * self.error * np.log(other),
            self.name,
            self.unit,
            self.fft_type,
        )

    def __len__(self):
        return len(self.value)

    def __lt__(self, other) -> bool:
        return self.value.max() < other.value.max()

    ####################################################################################################
    #                   Main Functions                                                                 #
    ####################################################################################################
    @property
    def value(self) -> np.ndarray:  # TODO: don't return array if input was not array
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

        elif isinstance(value, (list, tuple)):
            self._value = np.array(value)

        elif isinstance(value, np.ndarray):
            self._value = value

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
        if not hasattr(self, "_error"):
            self._error = (
                np.full(np.shape(self.value), np.nan)
                if self.value.size
                else np.empty(0)
            )

        if np.iscomplexobj(error):
            self._error.__setattr__("dtype", np.complex128)

        if isinstance(error, (float, np.integer, int)):
            self._error.fill(error)

        elif isinstance(error, (list, tuple, np.ndarray)):
            error = np.array(error)
            ndim = np.ndim(error)

            if ndim == 0:
                self._error.fill(error)

            elif ndim == 1:
                if np.iscomplexobj(error):
                    self._error = error
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
    def I(self):
        return self.real

    @property
    def Q(self):
        return self.imag

    @property
    def abs(self):
        return Valueclass(np.abs(self.value), np.abs(self.error), self.name, self.unit)

    @property
    def phase(self):
        return Valueclass(
            np.unwrap(np.angle(self.value)),
            np.array(np.angle(self.error)),
            self.name,
            "rad",
            self.fft_type,
        )

    @property
    def angle(self):
        return self.phase

    ####################################################################################################
    #                   Property Functions                                                             #
    ####################################################################################################
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

    @property
    def sprt(self):
        return Valueclass(
            np.sqrt(self.value),
            self.error / (2 * np.sqrt(self.value)),
            self.name,
            self.unit,
            self.fft_type,
        )

    ####################################################################################################
    #                   Math (Simple) Functions                                                        #
    ####################################################################################################
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

    def sum(self, axis=None):
        return np.sum(self.value, axis=axis)  # type: ignore

    def prod(self, axis=None):
        return np.prod(self.value, axis=axis)

    def clip(
        self,
        v_min=None,
        v_max=None,
        e_min=None,
        e_max=None,
        out_value=None,
        out_error=None,
        clip_value=True,
        clip_error=True,
    ):
        if not v_max:
            v_max = np.max(self.value)

        if not e_max:
            e_max = np.max(self.error)

        value = (
            np.clip(self.value, v_min, v_max, out_value) if clip_value else self.value
        )
        error = (
            np.clip(self.error, e_min, e_max, out_error) if clip_error else self.error
        )

        return Valueclass(
            value,
            error,
            self.name,
            self.unit,
            self.fft_type,
        )

    ####################################################################################################
    #                   Math (Advanced) Functions                                                      #
    ####################################################################################################
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
            self.value / np.sqrt(np.sum(self.value**2)),
            self.error / np.sqrt(np.sum(self.value**2)),
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

    @property
    def ddx(self):  # TODO: fix this
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

    def traces(self, operation="Show individual"):
        if operation in (
            "Show individual",
            "show individual",
            "individual",
            "Individual",
        ):
            return Valueclass(
                self.value,
                self.error,
                self.name,
                self.unit,
                self.fft_type,
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

            return Valueclass(
                v,
                self.error,
                self.name,
                self.unit,
                self.fft_type,
            )

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

    ####################################################################################################
    #                   Plot Functions                                                                 #
    ####################################################################################################
    def plot(self, *args, **kwargs):
        x = kwargs.pop("x", np.arange(len(self.value)))
        x_label = kwargs.pop("x_label", "index")
        title = kwargs.pop("title", None)
        fmt = kwargs.pop("fmt", ".")

        if np.iscomplexobj(self.value):
            self._plot_complex(x, x_label, title, fmt, *args, **kwargs)
        else:
            self._plot_1d(x, x_label, title, fmt, *args, **kwargs)

    def _plot_1d(self, x, x_label, title, fmt, *args, **kwargs):
        x = kwargs.pop("x", np.arange(len(self.value)))
        x_label = kwargs.pop("x_label", "index")

        y = kwargs.pop("y", self.value.real)
        y_label = kwargs.pop("y_label", self.name)

        title = kwargs.pop("title", None)
        fmt = kwargs.pop("fmt", ".")

        plt.errorbar(x, y, yerr=self.error, fmt=fmt, *args, **kwargs)
        plt.plot(x, y, *args, **kwargs)

        plt.xlabel(x_label)
        plt.ylabel(f"{y_label} [{self.unit}]")

        if title:
            plt.title(title)

        plt.show()

    def _plot_complex(self, x, x_label, title, fmt, *args, **kwargs):
        x = kwargs.pop("x", np.arange(len(self.value)))
        x_label = kwargs.pop("x_label", "index")

        y = kwargs.pop("y", self.value)
        y_label = kwargs.pop("y_label", self.name)
        y_real_label, y_imag_label = f"{y_label} real", f"{y_label} imag"

        title = kwargs.pop("title", None)
        fmt = kwargs.pop("fmt", ".")

        plt.errorbar(
            y.real,
            y.imag,
            yerr=self.error.real,
            xerr=self.error.imag,
            fmt=fmt,
            zorder=0,
            *args,
            **kwargs,
        )

        scatter_plot = plt.scatter(
            y.real,
            y.imag,
            c=x,
            *args,
            **kwargs,
        )

        plt.colorbar(scatter_plot, label=x_label)

        plt.xlabel(f"{y_real_label} [{self.unit}]")
        plt.ylabel(f"{y_imag_label} [{self.unit}]")

        if title:
            plt.title(title)

        plt.show()

    ####################################################################################################
    #                   Conversion Functions                                                           #
    ####################################################################################################
    def todict(self, split_complex: bool = False):
        """Converts the Parameter into a dictionary.

        Args:
            split_complex: Whether to split the value and error of a complex
                parameter into real and imaginary parts.
        """
        # Convert Parameter to a dictionary
        valuedict = asdict(self)

        # If the value of the Parameter is complex and split_complex is True
        if np.iscomplexobj(self.value) and split_complex:
            # Convert the value to a dictionary with real and imaginary parts
            valuedict["value"] = {
                "real": self.value.real.tolist(),
                "imag": self.value.imag.tolist(),
            }
        else:
            # Convert the value to a list
            valuedict["value"] = self.value.tolist()

        # If the error of the Parameter is complex and split_complex is True
        if np.iscomplexobj(self.error) and split_complex:
            # Convert the error to a dictionary with real and imaginary parts
            valuedict["error"] = {
                "real": self.error.real.tolist(),
                "imag": self.error.imag.tolist(),
            }
        else:
            # Convert the error to a list
            valuedict["error"] = self.error.tolist()

        # If the error is all NaNs
        if np.isnan(self.error).all():
            # Remove the error from the dictionary
            valuedict.pop("error")

        # Return the dictionary
        return valuedict

    def tostr(
        self, algin: bool = True, scale_values: bool = True, name_width=40, size_width=7
    ):
        """Converts Valueclass to a nice string, for printing. self.value and self.error are shown as number of points, minimum and maximum values."""

        def _getstr(self, scale_values: bool = True):
            value, unit_prefix, conversion_factor = self.value, "", 1
            if scale_values:
                value, unit_prefix, conversion_factor = convert_array_with_unit(
                    self.value
                )

            if self.value.size > 1:
                return f"{self.name}: {self.value.size}; {np.min(value)} - {np.max(value)} {unit_prefix}{self.unit}"

            value = (
                value[0]
                if np.isnan(self.error)
                else round_on_error(value[0], self.error[0] * conversion_factor)
            )
            return f"{self.name}: {value} {unit_prefix}{self.unit}"

        def _alginstr(vstr: str, algin: bool = True, name_width=40, size_width=7):
            if not algin:
                return vstr

            (name_str, vstr) = vstr.split(":") if ":" in vstr else ("", vstr)
            (size_str, vstr) = vstr.split(";") if ";" in vstr else ("", vstr)

            return f"{name_str : <{name_width}}{size_str : <{size_width}}{vstr}"

        vstr = _getstr(self, scale_values=scale_values)
        return _alginstr(vstr, algin, name_width, size_width)

    @staticmethod
    def fromdict(newdict: dict):
        def _get_numbers_from_dict(numbersdict):
            """Converts a dictionary of numbers to a 1-D complex array.
            The dictionary can contain either the keys "real" and "imag" or "I" and "Q"

            Args:
                numbersdict (dict): Dictionary of numbers.

            Returns:
                (np.ndarray): 1-D complex array of numbers.
            """
            # Check if the input is a dictionary.
            # If not, return the input as is.
            if not isinstance(numbersdict, dict):
                return numbersdict

            # Check if the dictionary has "real" and "imag" keys.
            # If so, extract the real and imaginary parts of the numbers.
            if "real" in numbersdict and "imag" in numbersdict:
                real = numbersdict["real"]
                imag = numbersdict["imag"]

            # Check if the dictionary has "I" and "Q" keys.
            # If so, extract the real and imaginary parts of the numbers.
            elif "I" in numbersdict and "Q" in numbersdict:
                real = numbersdict["I"]
                imag = numbersdict["Q"]

            # If the dictionary doesn't have the "real" and "imag" or "I" and "Q" keys,
            # return the input as is.
            else:
                return numbersdict

            # Convert the real and imaginary parts to numpy arrays and return them
            # as a complex array.
            return np.array(real) + 1j * np.array(imag)

        if "value" in newdict:
            newdict["value"] = _get_numbers_from_dict(newdict["value"])

        if "error" in newdict:
            newdict["error"] = _get_numbers_from_dict(newdict["error"])

        return Valueclass(**newdict)

    @staticmethod
    def fromfloat(
        value: Union["Valueclass", list, tuple, np.ndarray],
        name: str = "",
        unit: str = "",
    ) -> "Valueclass":
        """Converts a float to a Valueclass object.

        Args:
            value (float): The value to be converted.
            name (str): The name of the value.

        Returns:
            Valueclass: The converted value.
        """
        if isinstance(value, Valueclass):
            return value
        return Valueclass(name=name, unit=unit, value=value)

    ####################################################################################################
    #                   Conversion Functions                                                           #
    ####################################################################################################
    def append(
        self,
        other: Union["Valueclass", list, tuple, np.ndarray],
        axis: Optional[int] = None,
    ) -> None:
        """Appends a value to the value array.

        Args:
            other (Union[Valueclass, list, tuple, np.ndarray]): The value to be appended.
            axis (Optional[int], optional): The axis to append the value to. Defaults to None.
        """
        if not isinstance(other, Valueclass):
            other = self.fromfloat(other, self.name)

        self._value = np.append(self.value, other.value, axis=axis)
        self._error = np.append(self.error, other.error, axis=axis)
