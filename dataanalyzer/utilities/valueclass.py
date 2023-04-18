"""Valueclass class for storing values and errors.

The class is designed to be used in a similar way as the numpy.ndarray class.
It can be sliced, added, subtracted, multiplied and divided with other Valueclass objects or with floats.
It uses some of the same functionality as seen in the Labber GUI (See .traces() method).
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional, Union, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pybaselines import Baseline

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
        return f"{self.name}:\n(value={self.value}, error={error}, unit={self.unit})\n\n"

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
        selfcopy = self.copy()
        if "Valueclass" not in str(type(other)):
            selfcopy.value += other
        else:
            selfcopy.value += other.value
            selfcopy.error = np.sqrt(self.error**2 + other.error**2)
        return selfcopy

    def __sub__(self, other) -> "Valueclass":
        """Subtracts two Valueclass objects or a Valueclass object and a float.

        Args:
            other (object, float, int): Object to subtract.

        Returns:
            Valueclass: Difference of the two objects.
        """
        selfcopy = self.copy()
        if "Valueclass" not in str(type(other)):
            selfcopy.value -= other
        else:
            selfcopy.value -= other.value
            selfcopy.error = np.sqrt(self.error**2 + other.error**2)
        return selfcopy

    def __mul__(self, other) -> "Valueclass":
        selfcopy = self.copy()
        if "Valueclass" not in str(type(other)):
            selfcopy.value *= other
            selfcopy.error *= other
        else:
            return Valueclass(
                self.value * other.value,
                np.sqrt((self.error * other.value) ** 2 + (self.value * other.error) ** 2),
                self.name,
                self.unit,
            )
        return selfcopy

    def __truediv__(self, other) -> "Valueclass":
        selfcopy = self.copy()
        if "Valueclass" not in str(type(other)):
            selfcopy.value /= other
            selfcopy.error /= other
        else:
            return Valueclass(
                self.value / other.value,
                np.sqrt((self.error / other.value) ** 2 + (self.value * other.error / other.value**2) ** 2),
                self.name,
                self.unit,
            )
        return selfcopy

    def __pow__(self, other) -> "Valueclass":
        selfcopy = self.copy()
        if "Valueclass" not in str(type(other)):
            return Valueclass(
                self.value**other,
                self.error * other * self.value ** (other - 1),  # TODO: make correct error propagation
                self.name,
                self.unit,
            )
        else:
            return Valueclass(
                self.value**other.value,
                np.sqrt(
                    (self.error * other.value * self.value ** (other.value - 1)) ** 2
                    + (self.value**other.value * other.error * np.log(self.value)) ** 2
                ),
                self.name,
                self.unit,
            )
        return selfcopy

    def __radd__(self, other) -> "Valueclass":
        return self + other

    def __rsub__(self, other) -> "Valueclass":
        return other - self

    def __rmul__(self, other) -> "Valueclass":
        return self * other

    def __rtruediv__(self, other) -> "Valueclass":
        return self / other

    def __rpow__(self, other) -> "Valueclass":
        return self**other

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
        if not hasattr(self, "_error") or self._error.shape != np.shape(self.value):
            self._error = np.full(np.shape(self.value), np.nan) if self.value.size else np.empty(0)

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

    @property  # TODO: Hej Malthe, denne funktion er rar at have, hvis du har et godt sted til den kan du godt flytte den. mvh Jacob
    def has_unit(self):
        return self.unit != ""

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
        return Valueclass(self.value.T, self.error.T, self.name, self.unit, self.fft_type)

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
    def mean(self, axis=None) -> "Valueclass":
        copy = self.copy()
        copy.value = np.mean(self.value, axis=axis)
        copy.error = np.std(self.value, axis=axis)
        return copy

    def std(self, axis=None) -> "Valueclass":
        copy = self.copy()
        copy.value = np.std(self.value, axis=axis)
        copy.error = np.std(self.error, axis=axis)
        return copy

    def min(self, axis=None) -> np.ndarray:
        return np.min(self.value, axis=axis)

    def max(self, axis=None) -> np.ndarray:
        return np.max(self.value, axis=axis)

    def argmin(self, axis=None) -> np.intp:
        return np.argmin(self.value, axis=axis)

    def argmax(self, axis=None) -> np.intp:
        return np.argmax(self.value, axis=axis)

    def min_error(self, axis=None) -> np.ndarray:
        return np.min(self.error, axis=axis)

    def max_error(self, axis=None) -> np.ndarray:
        return np.max(self.error, axis=axis)

    def argmin_error(self, axis=None) -> np.intp:
        return np.argmin(self.error, axis=axis)

    def argmax_error(self, axis=None) -> np.intp:
        return np.argmax(self.error, axis=axis)

    def sum(self, axis=None) -> np.ndarray:
        return np.sum(self.value, axis=axis)  # type: ignore

    def prod(self, axis=None) -> np.ndarray:
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
    ) -> "Valueclass":
        v_max = v_max or np.max(self.value)
        e_max = e_max or np.max(self.error)

        value = np.clip(self.value, v_min, v_max, out_value) if clip_value else self.value
        error = np.clip(self.error, e_min, e_max, out_error) if clip_error else self.error

        return Valueclass(
            value,
            error,
            self.name,
            self.unit,
            self.fft_type,
        )
        copy.error = np.clip(self.error, e_min, e_max, out_error) if clip_error else self.error
        return copy

    def round(self, decimals=0, out_value=None, out_error=None) -> "Valueclass":
        copy = self.copy()
        copy.value = np.round(self.value, decimals, out_value)
        copy.error = np.round(self.error, decimals, out_error)
        return copy

    def sort(self, axis=-1, sort_by="value", kind=None, order=None, reverse=False) -> "Valueclass":
        sign = -1 if reverse else 1

        if sort_by == "value":
            return self[np.argsort(sign * self.value, axis=axis, kind=kind, order=order)]
        elif sort_by == "error":
            return self[np.argsort(sign * self.error, axis=axis, kind=kind, order=order)]
        else:
            raise ValueError("sort_by must be either 'value' or 'error'")

    def delete(self, obj, axis=None) -> "Valueclass":
        copy = self.copy()
        copy.value = np.delete(self.value, obj, axis)
        copy.error = np.delete(self.error, obj, axis)
        return copy

    ####################################################################################################
    #                   Filter Functions                                                               #
    ####################################################################################################
    def savgol(
        self,
        x=None,
        peak_width=None,
        N=3,
        return_baseline=False,
        average=False,
        axis=0,
        **kwargs,
    ):
        if isinstance(x, Valueclass):
            x = x.value

        x = np.arange(self.value.size) if x is None else np.array(x)

        if peak_width is None:
            peak_width = (np.max(x) - np.min(x)) / 15

        sigma_factor = kwargs.pop("sigma_factor", 10)
        sigma = kwargs.pop("sigma", peak_width * sigma_factor)

        filter_func = self._savgol_filter_1d
        return self._genereal_filer_remover(
            filter_func=filter_func,
            x=x,
            sigma=sigma,
            N=N,
            axis=axis,
            return_baseline=return_baseline,
            average=average,
            **kwargs,
        )

    def _savgol_filter_1d(self, x, y, sigma, N, x_output=None):
        """Applies a Savitzky–Golay filter [1] to the input data, fitting a weighted degree-N
        polynomial [2] with a Gaussian weighing function with standard deviation sigma around each point.
        If x_out is specified the data is interpolated to the points of x_out.
        [1] https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
        [2] https://en.wikipedia.org/wiki/Weighted_least_squares
        """

        def _gaussian_filter(x, sigma):
            return np.exp(-(x**2) / (2 * sigma**2))

        if x_output is None:
            x_output = x

        xdiff_list = [x - x_out_i for x_out_i in x_output]
        X_list = [np.vander(xdiff, N=N, increasing=True) for xdiff in xdiff_list]
        w_list = [_gaussian_filter(x, sigma=sigma) for x in xdiff_list]
        Xp_list = [np.diag(w) @ X for w, X in zip(w_list, X_list)]
        yp_list = [w * y for w in w_list]
        Xp_inv_list = [np.linalg.pinv(Xp) for Xp in Xp_list]
        return np.array([Xp_inv_list[i][0] @ yp_list[i] for i in range(len(yp_list))])

    def savgol_removed_outliers(self, x, x_output=None, **kwargs):
        """Removes outliers from the data using a Savitzky–Golay filter"""
        max_iterations = kwargs.pop("max_iterations", 10)
        if isinstance(x, Valueclass):
            x = x.value
        if x_output is None:
            x_output = x
        dx = x[1] - x[0]
        width = dx / 4

        converged = False
        mask = np.ones_like(x, dtype=bool)
        iteration = 0
        while not (converged and iteration < max_iterations):
            iteration += 1
            filtered = self[mask].savgol(x[mask], peak_width=width, return_baseline=True, x_output=x)
            baseline_removed = self - filtered
            new_mask = baseline_removed.remove_outliers(return_mask=True)
            if np.all(new_mask == mask):
                converged = True
            mask = new_mask

        filtered_output = self[mask].savgol(x[mask], peak_width=width, return_baseline=True, x_output=x_output)
        return filtered_output, mask

    def remove_baseline(
        self,
        baseline_type="modpoly",
        return_baseline=False,
        average=False,
        axis=0,
        **kwargs,
    ) -> "Valueclass":
        kwargs.setdefault("axis_name", "(Baseline)")
        filter_func = self._remove_baseline_1d

        return self._genereal_filer_remover(
            filter_func,
            baseline_type=baseline_type,
            return_baseline=return_baseline,
            average=average,
            axis=axis,
            **kwargs,
        )

    def _remove_baseline_1d(self, x, y, baseline_type, **kwargs):
        baseline_fitter = Baseline(x_data=x)

        if baseline_type == "modpoly":
            bkg = baseline_fitter.modpoly(y, **kwargs)
        elif baseline_type == "asls":
            bkg = baseline_fitter.asls(y, **kwargs)
        elif baseline_type == "mor":
            bkg = baseline_fitter.mor(y, **kwargs)
        elif baseline_type == "snip":
            bkg = baseline_fitter.snip(y, **kwargs)
        else:
            raise ValueError("Unknown baseline type. Choose from 'modpoly', 'asls', 'mor' or 'snip'.")

        return bkg[0]

    def _genereal_filer_remover(
        self,
        filter_func,
        x=None,
        axis=0,
        return_baseline=False,
        average=False,
        **kwargs,
    ) -> "Valueclass":
        axis_name = kwargs.pop("axis_name", "(Filtered)")

        if self.value.ndim == 1:
            x = np.arange(len(self.value)) if x is None else x
            bkg = filter_func(x=x, y=self.value, **kwargs)

        else:
            y = self.value.T if axis == 1 else self.value
            x = kwargs.pop("x", np.arange(y.shape[0]))

            if average:
                bkg_one = filter_func(x=x, y=np.mean(y, axis=1), **kwargs)
                bkg = np.vstack([bkg_one] * y.shape[1]).T
            else:
                bkg = np.zeros(y.shape)
                for i in range(y.shape[1]):
                    bkg[:, i] = filter_func(x=x, y=y[:, i], **kwargs)

            if axis == 1:
                bkg = bkg.T

        self_copy = self.copy()

        if return_baseline:
            return Valueclass(bkg, name=self.name + f" {axis_name}", unit=self.unit)

        self_copy.value -= bkg
        return self_copy

    def remove_outliers(self, sigma=3, axis=None, return_mask=False, converge=True):
        mask = np.abs(self.value - np.mean(self.value, axis=axis)) < sigma * np.std(self.value, axis=axis)

        if converge:
            mask_temp = np.ones_like(mask)
            while not np.all(mask == mask_temp):
                mask_temp = mask
                mask = np.abs(self.value - np.mean(self.value[mask])) < sigma * np.std(self.value[mask])

        return mask if return_mask else self[mask]

    ####################################################################################################
    #                   Math (Advanced) Functions                                                      #
    ####################################################################################################
    @property
    def db(self) -> "Valueclass":
        copy = self.copy()
        copy.value = 20 * np.log10(self.value)
        copy.error = 20 * self.error / (np.log(10) * self.value)
        copy.unit = "dB"
        return copy

    @property
    def norm(self) -> "Valueclass":
        copy = self.copy()
        copy.value = self.value / np.sqrt(np.sum(self.value**2))
        copy.error = self.error / np.sqrt(np.sum(self.value**2))
        return copy

    @property
    def fft(self) -> "Valueclass":
        fft = np.fft.fft(self.value)
        N = np.size(self.value)

        copy = self.copy()
        copy.value = 2.0 / N * np.abs(fft[: N // 2])
        copy.error = np.nan
        copy.name = f"FFT of {self.name}"
        copy.fft_type = "fft_y"
        return copy

    @property
    def fftfreq(self) -> "Valueclass":
        N = np.size(self.value)

        copy = self.copy()
        copy.value = np.fft.fftfreq(N, d=self.value[1] - self.value[0])[: N // 2]
        copy.error = np.nan
        copy.name = f"FFT of {self.name}"
        copy.fft_type = "fft_x"
        return copy

    @property
    def substract_mean(self):
        return Valueclass(self.value - np.mean(self.value), self.error, self.name, self.unit)

    @property
    def ddx(self) -> "Valueclass":  # TODO: Chack if this is correct
        copy = self.copy()
        copy.value = np.gradient(self.value)
        copy.error = np.gradient(self.error)
        return copy

    @property
    def ddxx(self) -> "Valueclass":
        return self.ddx.ddx

    def norm_zero_to_one(self, axis=None):
        return Valueclass(
            (self.value - np.min(self.value, axis=axis))
            / (np.max(self.value, axis=axis) - np.min(self.value, axis=axis)),
            self.error / (np.max(self.value, axis=axis) - np.min(self.value, axis=axis)),
            self.name,
            self.unit,
            self.fft_type,
        )

        copy = self.copy()
        copy.value = (self.value - np.min(self.value, axis=axis)) / scale
        copy.error = self.error / scale
        return copy

    def traces(self, operation="Show individual") -> "Valueclass":
        copy = self.copy()

        if operation == "substract first":
            copy.value = self.value - self.value[0]

        elif operation == "substract mean":
            copy.value = self.value - np.mean(self.value, axis=0)

        elif operation == "substract last":
            copy.value = self.value - self.value[-1]

        elif operation == "substract min":
            copy.value = self.value - np.min(self.value, axis=0)

        elif operation == "substract max":
            copy.value = self.value - np.max(self.value, axis=0)

        elif operation == "substract median":
            copy.value = self.value - np.median(self.value, axis=0)

        elif operation == "substract previous":
            value = self.value - np.roll(self.value, 1, axis=0)  # type: ignore
            value[0] = np.zeros(self.value.shape[1])
            copy.value = value

        elif operation == "average":
            copy.value = np.tile(np.mean(self.value, axis=0), (np.shape(self.value)[0], 1))
            copy.error = np.tile(np.mean(self.error, axis=0), (np.shape(self.value)[0], 1))

        elif operation == "standard deviation":
            copy.value = np.tile(np.std(self.value, axis=0), (np.shape(self.value)[0], 1))
            copy.error = np.tile(np.std(self.error, axis=0), (np.shape(self.value)[0], 1))

        else:
            raise ValueError(
                f"Operation '{operation}' not recognized\n"
                "self.traces() takes one of the following arguments:"
                "'substract first', 'substract mean',"
                "'substract last', 'substract min', 'substract max', 'substract median',"
                "'substract previous', 'average', 'standard deviation'"
            )

        return copy

    def gradiant(self, axis=None) -> "Valueclass":
        copy = self.copy()
        copy.value = np.gradient(self.value, axis=axis)
        return copy

    ####################################################################################################
    #                   Plot Functions                                                                 #
    ####################################################################################################
    def plot(self, *args, **kwargs) -> None:
        x = kwargs.pop("x", np.arange(len(self.value)))
        x_label = kwargs.pop("x_label", "index")
        title = kwargs.pop("title", None)
        fmt = kwargs.pop("fmt", ".")

        if np.iscomplexobj(self.value):
            self._plot_complex(x, x_label, title, fmt, *args, **kwargs)
        elif np.ndim(self.value) == 2:
            self._plot_2d(x, x_label, title, *args, **kwargs)
        else:
            self._plot_1d(x, x_label, title, fmt, *args, **kwargs)

    def _plot_1d(self, x, x_label, title, fmt, *args, **kwargs) -> None:
        y = kwargs.pop("y", self.value.real)
        y_label = kwargs.pop("y_label", self.name)

        plt.errorbar(x, y, yerr=self.error, fmt=fmt, *args, **kwargs)
        plt.plot(x, y, *args, **kwargs)

        plt.xlabel(x_label)
        plt.ylabel(f"{y_label} [{self.unit}]")

        if title:
            plt.title(title)

        plt.show()

    def _plot_2d(self, x, x_label, title, *args, **kwargs) -> None:
        y = kwargs.pop("y", np.arange(len(self.value[1])))
        y_label = kwargs.pop("y_label", "index 1")
        colorbar = kwargs.pop("colorbar", True)

        fig, ax = plt.subplots()
        ax.grid(False)
        c = ax.pcolormesh(y, x, self.value, *args, **kwargs)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(False)

        if colorbar:
            fig.colorbar(c, ax=ax)

        if title:
            plt.title(title)

        plt.show()

    def _plot_complex(self, x, x_label, title, fmt, *args, **kwargs) -> None:
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
    def todict(self, split_complex: bool = False) -> Dict[str, Any]:
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
        self,
        algin: bool = True,
        scale_values: bool = True,
        name_width=30,
        size_width=7,
        decimals=2,
        string_type="metadata",
    ):
        """Converts Valueclass to a nice string, for printing. self.value and self.error are shown as number of points, minimum and maximum values."""

        def _getstr(self, scale_values: bool = True, decimals: int = 2):
            if self.value.dtype == int:
                decimals = 0

            value, unit_prefix, conversion_factor = self.value, "", 1
            if scale_values and self.has_unit:
                value, unit_prefix, conversion_factor = convert_array_with_unit(self.value)

            if self.value.size > 1:
                return f"{self.name}: ({self.value.size}); {np.min(value):.{decimals}f}–{np.max(value):.{decimals}f} {unit_prefix}{self.unit}"

            value = value[0] if np.isnan(self.error) else round_on_error(value[0], self.error[0] * conversion_factor)
            return f"{self.name}: {value:.{decimals}f} {unit_prefix}{self.unit}"

        def _alginstr(vstr: str, algin: bool = True, name_width=40, size_width=7):
            if not algin:
                return vstr

            (name_str, vstr) = vstr.split(":") if ":" in vstr else ("", vstr)
            (size_str, vstr) = vstr.split(";") if ";" in vstr else ("", vstr)

            return f"{name_str : <{name_width}}{vstr}{size_str : <{size_width}}"

        def _getstr_value(self, scale_values: bool = True, decimals: int = 2):
            # string for printing only the value of the parameter
            value, unit_prefix, conversion_factor = self.value, "", 1
            if scale_values and self.has_unit:
                value, unit_prefix, conversion_factor = convert_array_with_unit(self.value)
            value = value[0] if np.isnan(self.error) else round_on_error(value[0], self.error[0] * conversion_factor)
            return f"{value:.{decimals}f} {unit_prefix}{self.unit}"

        if string_type == "metadata":
            vstr = _getstr(self, scale_values=scale_values, decimals=decimals)
            return _alginstr(vstr, algin, name_width, size_width)
        elif string_type == "value":
            return _getstr_value(self, scale_values=scale_values, decimals=decimals)

    @staticmethod
    def fromdict(newdict: dict) -> "Valueclass":
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
            return value.copy()
        return Valueclass(name=name, unit=unit, value=value)

    @staticmethod
    def fromlisttoone(
        newlist: list[dict],
        name: Optional[str] = None,
        unit: Optional[str] = None,
        **kwargs,
    ) -> "Valueclass":
        """Converts a list of dictionaries to a Valueclass object.

        Args:
            newlist (list[dict]): The list of dictionaries to be converted.

        Returns:
            Valueclass: The converted list.
        """
        newdict = {}
        for key in newlist[0].keys():
            newdict[key] = [newdict[key] for newdict in newlist]

        if name is not None:
            newdict["name"] = name

        if unit is not None:
            newdict["unit"] = unit

        if kwargs.get("value", None):
            newdict["value"] = kwargs["value"]

        if kwargs.get("error", None):
            newdict["error"] = kwargs["error"]

        if kwargs.get("fft_type", None):
            newdict["fft_type"] = kwargs["fft_type"]

        if kwargs.get("sweep_idx", None):
            newdict["sweep_idx"] = kwargs["sweep_idx"]

        return Valueclass.fromdict(newdict)

    @staticmethod
    def fromdf(
        df: Union[pd.DataFrame, pd.Series],
        name: str = "",
        unit: str = "",
        value_col: str = "value",
        error_col: str = "error",
        **kwargs,
    ) -> "Valueclass":
        """Converts a pandas dataframe to a Valueclass object.

        Args:
            df (pd.DataFrame): The dataframe to be converted.
            name (str): The name of the value.
            unit (str): The unit of the value.

        Returns:
            Valueclass: The converted dataframe.
        """
        if isinstance(df, pd.DataFrame):
            value = df[value_col].to_numpy() if value_col in df.columns else df.to_numpy()
            error = df[error_col].to_numpy() if error_col in df.columns else np.zeros_like(value)

        elif isinstance(df, pd.Series):
            value = df.to_numpy()
            error = np.zeros_like(value)

        return Valueclass(
            name=name,
            unit=unit,
            value=value,
            error=error,
            **kwargs,
        )

    def copy(self) -> "Valueclass":
        return deepcopy(self)

    def astype(self, dtype, **kwargs):
        self.value = self.value.astype(dtype, **kwargs)
        return self

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

        return self


if __name__ == "__main__":
    # Test the Valueclass class. (Mean fucntion)
    import numpy as np

    test = Valueclass(name="test", unit="V", value=y)
    test.plot()
    test.remove_outliers(sigma=3, converge=True).plot()

    test = Valueclass(name="test", unit="V", value=y)
