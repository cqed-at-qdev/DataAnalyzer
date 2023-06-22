# Author: Malthe Asmus Marciniak Nielsen
from typing import Any, Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd

from dataanalyzer.utilities import Valueclass


def matplotlib_decorator(
    func: Callable[..., Any]
):  # TODO: don't overwrite axis labels if they are already set and new data don't have name/units
    """Decorator for matplotlib functions to make them return a Valueclass object.

    Args:
        func: The function to be decorated.
    """

    def _matplotlib_general(self, ax=None, **kwargs) -> tuple[Valueclass, Valueclass, Optional[Valueclass], dict]:
        """Wrapper for matplotlib functions to make them return a Valueclass object."""

        if "cycle_color" in kwargs:
            mpl_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            kwargs["color"] = mpl_cycle[kwargs.pop("cycle_color") % len(mpl_cycle)]

        if isinstance(ax, plt.Axes):
            self.ax = ax

        elif isinstance(ax, int):
            self.ax = self.axs.flatten()[ax]
        else:
            self.ax = self.axs[ax] if ax else self.ax

        x, y, z, xlabel, ylabel, title = _set_all_data(kwargs)

        self.ax.set_title(title)

        if kwargs.pop("overwrite", False):
            self.ax.clear()
            self.ax.set_prop_cycle(None)

        self.ax.set_xlabel(xlabel)

        self.ax.set_ylabel(ylabel)

        self.metadata += kwargs.pop("metadata", "")

        return x, y, z, kwargs

    def _set_all_data(kwargs: dict[str, Any]):
        x, xlabel = _set_data("x", kwargs)
        y, ylabel = _set_data("y", kwargs)
        z = _set_zdata("z", kwargs)

        x, y = _check_and_update_fft(x, y)

        title = kwargs.pop("title", "")

        return x, y, z, xlabel, ylabel, title

    def _set_data(data_name: str, kwargs: dict[str, Any]) -> tuple[Valueclass, str]:
        data_float: Union[list, tuple, np.ndarray] = kwargs.pop(data_name, None)

        # Convert the data to a Valueclass instance
        if data_float is not None:
            if isinstance(data_float, pd.Series):
                data_float = data_float.values
            data = Valueclass.fromfloat(data_float, f"{data_name} data")
            default_label = f"{data.name} [{data.unit}]" if data.unit else data.name
        else:
            data = None
            default_label = ""

        # Set the data label
        label = kwargs.pop(f"{data_name}label", default_label)

        # Return the data and the label
        return data, label

    def _set_zdata(data_name: str, kwargs: dict[str, Any]) -> Optional[Valueclass]:
        data_float: Union[list, tuple, np.ndarray] = kwargs.pop(data_name, None)

        if data_float is not None:
            return Valueclass.fromfloat(data_float, f"{data_name} data")
        return None

    def _check_and_update_fft(x: Valueclass, y: Valueclass) -> tuple[Valueclass, Valueclass]:
        """Checks if the x and y data are fft or fftfreq data and updates them accordingly.

        Args:
            x (Valueclass): x data to be checked.
            y (Valueclass): y data to be checked.

        Raises:
            ValueError: If the x and y data are both fft or fftfreq data.

        Returns:
            tuple[Valueclass, Valueclass]: The updated x and y data.
        """
        if not x or not y:
            return x, y

        if not x.fft_type and not y.fft_type:
            return x, y

        if x.fft_type == y.fft_type:
            raise ValueError(f"The x and y data cannot both be same fft_type ({x.fft_type}).")

        if not x.fft_type:
            x = copy.deepcopy(x.fftfreq) if y.fft_type == "fft_y" else copy.deepcopy(x.fft)

        elif not y.fft_type:
            y = copy.deepcopy(y.fftfreq) if x.fft_type == "fft_y" else copy.deepcopy(y.fft)

        return x, y

    def _plot_legends(self):
        """Adds legends to the plot if the user has specified them."""
        for ax in self.axs.flatten():
            if ax.get_legend_handles_labels() != ([], []) and not hasattr(ax, "bloch"):
                ax.legend()  # bbox_to_anchor=(1.04, 1), loc="upper left"

            if hasattr(ax, "subax"):
                if ax.subax.get_legend_handles_labels() != ([], []):
                    ax.subax.legend()  # bbox_to_anchor=(1.04, 1), loc="upper left"

    def wrapper(
        self,
        x: Union[Valueclass, list, tuple, np.ndarray],
        y: Optional[Union[Valueclass, list, tuple, np.ndarray]] = None,
        z: Optional[Union[Valueclass, list, tuple, np.ndarray]] = None,
        ax: Union[tuple, int] = (),
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

        if not y:
            func(self, x=x, ax=ax, *args, **kwargs)
            _plot_legends(self)
            return

        if z:
            self.ax.grid(False)
            func(self, x=x, y=y, z=z, ax=ax, *args, **kwargs)
            _plot_legends(self)
            return

        func(self, x=x, y=y, ax=ax, *args, **kwargs)
        _plot_legends(self)

    return wrapper
