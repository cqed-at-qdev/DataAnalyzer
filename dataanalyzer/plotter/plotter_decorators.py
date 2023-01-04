# Author: Malthe Asmus Marciniak Nielsen
import contextlib
from typing import Any, Callable, Optional, Union

import numpy as np
import copy

from dataanalyzer.utilities import Valueclass, from_float_to_valueclass


def matplotlib_decorator(func: Callable[..., Any]):
    """Decorator for matplotlib functions to make them return a Valueclass object.

    Args:
        func: The function to be decorated.
    """

    def _matplotlib_general(
        self, ax=None, **kwargs
    ) -> tuple[Valueclass, Valueclass, Optional[Valueclass], dict]:
        """Wrapper for matplotlib functions to make them return a Valueclass object."""

        self.ax = self.axs[ax] if ax else self.ax

        x, y, z, xlabel, ylabel, title = _set_all_data(kwargs)

        self.ax.set_title(title)

        overwrite = kwargs.pop("overwrite", False)
        if overwrite:
            self.ax.clear()
            self.ax.set_prop_cycle(None)

        with contextlib.suppress(TypeError):
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)

        self.metadata += kwargs.pop("metadata", "")

        return x, y, z, kwargs

    def _set_all_data(kwargs: dict[str, Any]):
        x, xlabel = _set_data("x", kwargs)
        y, ylabel = _set_data("y", kwargs)
        z = _set_data("z", kwargs) if "z" in kwargs else None

        x, y = _check_and_update_fft(x, y)

        title = kwargs.pop("title", f"{y.name} vs {x.name}")

        return x, y, z, xlabel, ylabel, title

    def _set_data(data_name: str, kwargs: dict[str, Any]):
        dn = data_name
        data_float: Union[list, tuple, np.ndarray] = kwargs.pop(dn, None)

        if data_float is None:
            return None

        data = from_float_to_valueclass(data_float, f"{dn} data")

        if dn != "z":
            default_label = f"{data.name} [{data.unit}]" if data.unit else data.name
            label = kwargs.pop(f"{dn}label", default_label)

            return data, label
        return data

    def _check_and_update_fft(x: Valueclass, y: Valueclass):
        """Checks if the x and y data is fft or not and sets the fft_type accordingly."""
        if y.fft_type == "fft_y" and x.fft_type != "fft_x":
            x = copy.deepcopy(x.fftfreq)
        elif y.fft_type == "fft_x" and x.fft_type != "fft_y":
            x = copy.deepcopy(x.fft)
        elif x.fft_type == "fft_x" and y.fft_type != "fft_y":
            y = copy.deepcopy(y.fftfreq)
        elif x.fft_type == "fft_y" and y.fft_type != "fft_x":
            y = copy.deepcopy(y.fft)

        return x, y

    def _plot_legends(self):
        """Adds legends to the plot if the user has specified them."""
        [
            ax.legend()
            for ax in self.axs.flatten()
            if ax.get_legend_handles_labels() != ([], [])
        ]

    def wrapper(
        self,
        x: Union[Valueclass, list, tuple, np.ndarray],
        y: Union[Valueclass, list, tuple, np.ndarray],
        z: Optional[Union[Valueclass, list, tuple, np.ndarray]] = None,
        ax: tuple = (),
        *args,
        **kwargs,
    ):
        """Wrapper for matplotlib functions to make them return a Valueclass object.

        Args:
            x (Valueclass): The x data.
            y (Valueclass): The y data.
            z (Valueclass, optional): The z data. Defaults to None.
            ax (tuple, optional): The axes to plot on. Defaults to ().
        """
        x, y, z, kwargs = _matplotlib_general(self, x=x, y=y, z=z, ax=ax, **kwargs)

        if x is None or y is None:
            raise ValueError("x and y data must be specified.")

        if z:
            self.ax.grid(False)
            func(self, x=x, y=y, z=z, ax=ax, *args, **kwargs)
            _plot_legends(self)
            return

        func(self, x=x, y=y, ax=ax, *args, **kwargs)
        _plot_legends(self)

    return wrapper
