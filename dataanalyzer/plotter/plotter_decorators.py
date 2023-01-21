# Author: Malthe Asmus Marciniak Nielsen
from typing import Any, Callable, Optional, Union

import numpy as np
import copy

from dataanalyzer.utilities import Valueclass


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

        title = kwargs.pop("title", f"{y.name} vs {x.name}")

        return x, y, z, xlabel, ylabel, title

    def _set_data(data_name: str, kwargs: dict[str, Any]) -> tuple[Valueclass, str]:
        # Pop the data from the keyword arguments
        data_float: Union[list, tuple, np.ndarray] = kwargs.pop(data_name, None)

        # If no data was provided, raise an error
        if data_float is None:
            raise ValueError(f"{data_name} data must be specified.")

        # Convert the data to a Valueclass instance
        data = Valueclass.fromfloat(data_float, f"{data_name} data")
        # Set the default label for the data
        default_label = f"{data.name} [{data.unit}]" if data.unit else data.name
        # Pop the label from the keyword arguments, if not found use the default label
        label = kwargs.pop(f"{data_name}label", default_label)

        # Return the data and the label
        return data, label

    def _set_zdata(data_name: str, kwargs: dict[str, Any]) -> Optional[Valueclass]:
        data_float: Union[list, tuple, np.ndarray] = kwargs.pop(data_name, None)

        if data_float is not None:
            return Valueclass.fromfloat(data_float, f"{data_name} data")
        return None

    def _check_and_update_fft(
        x: Valueclass, y: Valueclass
    ) -> tuple[Valueclass, Valueclass]:
        """Checks if the x and y data are fft or fftfreq data and updates them accordingly.

        Args:
            x (Valueclass): x data to be checked.
            y (Valueclass): y data to be checked.

        Raises:
            ValueError: If the x and y data are both fft or fftfreq data.

        Returns:
            tuple[Valueclass, Valueclass]: The updated x and y data.
        """
        if not x.fft_type and not y.fft_type:
            return x, y

        if x.fft_type == y.fft_type:
            raise ValueError(
                f"The x and y data cannot both be same fft_type ({x.fft_type})."
            )

        if not x.fft_type:
            x = (
                copy.deepcopy(x.fftfreq)
                if y.fft_type == "fft_y"
                else copy.deepcopy(x.fft)
            )

        elif not y.fft_type:
            y = (
                copy.deepcopy(y.fftfreq)
                if x.fft_type == "fft_y"
                else copy.deepcopy(y.fft)
            )

        return x, y

    def _plot_legends(self):
        """Adds legends to the plot if the user has specified them."""
        for ax in self.axs.flatten():
            if ax.get_legend_handles_labels() != ([], []):
                ax.legend()

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
