# Author: Malthe Asmus Marciniak Nielsen
from typing import Any, Callable, Optional, Union

import numpy as np
import copy

from dataanalyzer.utilities import Valueclass, from_float_to_valueclass


def matplotlib_decorator(func: Callable[..., Any]):
    """Decorator for matplotlib functions to make them return a Valueclass object.

    Args:
        func: The function to be decorated.
    """

    def _matplotlib_genereal(
        self, ax=None, **kwargs
    ) -> tuple[Valueclass, Valueclass, Optional[Valueclass], dict]:
        """Wrapper for matplotlib functions to make them return a Valueclass object."""

        if ax:
            self.ax = self.axs[ax]

        _x: Union[list, tuple, np.ndarray] = kwargs.pop("x", None)
        _y: Union[list, tuple, np.ndarray] = kwargs.pop("y", None)
        _z: Union[list, tuple, np.ndarray] = kwargs.pop("z", None)

        x = from_float_to_valueclass(_x, "x data")
        y = from_float_to_valueclass(_y, "y data")
        z = from_float_to_valueclass(_z, "z data") if _z is not None else None

        if y.fft_type == "fft_y" and x.fft_type != "fft_x":
            x = copy.deepcopy(x.fftfreq)
        elif y.fft_type == "fft_x" and x.fft_type != "fft_y":
            x = copy.deepcopy(x.fft)
        elif x.fft_type == "fft_x" and y.fft_type != "fft_y":
            y = copy.deepcopy(y.fftfreq)
        elif x.fft_type == "fft_y" and y.fft_type != "fft_x":
            y = copy.deepcopy(y.fft)

        xlabel = kwargs.pop("xlabel", f"{x.name} [{x.unit}]" if x.unit else x.name)
        if xlabel != "x data":
            self.ax.set_xlabel(xlabel)

        ylabel = kwargs.pop("ylabel", f"{y.name} [{y.unit}]" if y.unit else y.name)
        if ylabel != "y data":
            self.ax.set_ylabel(ylabel)

        z = from_float_to_valueclass(_z, "z data") if _z is not None else None

        if x and y and x.name != "x data" and y.name != "y data":
            title = kwargs.pop("title", f"{x.name} vs {y.name}")
            self.ax.set_title(title)

        self.metadata += kwargs.pop("metadata", "")

        return x, y, z, kwargs

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
        x, y, z, kwargs = _matplotlib_genereal(self, x=x, y=y, z=z, ax=ax, **kwargs)

        if kwargs.get("title"):
            self.ax.set_title(kwargs.pop("title"))

        if z:
            grid = kwargs.pop("grid", False)
            self.ax.grid(grid)
            func(self, x=x, y=y, z=z, ax=ax, *args, **kwargs)
            _plot_legends(self)
            return

        func(self, x=x, y=y, ax=ax, *args, **kwargs)
        _plot_legends(self)

    return wrapper
