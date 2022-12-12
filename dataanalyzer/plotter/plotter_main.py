# Author: Malthe Asmus Marciniak Nielsen
import contextlib
import os
from typing import Union
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

from dataanalyzer.plotter.plotter_decorators import matplotlib_decorator
from dataanalyzer.utilities import Valueclass
from dataanalyzer.utilities.utilities import convert_array_with_unit


class Plotter:
    def __init__(
        self,
        subplots=(1, 1),
        default_settings: Union[dict, str, bool] = True,
        **kwargs,
    ):
        """The Plotter class is a wrapper for matplotlib.pyplot. It is used to plot data in a consistent way.

        Args:
            subplots (tuple, optional): The shape of the subplots. Defaults to (1, 1).
        """
        self.set_default_settings(default_settings)

        self.kwargs = kwargs

        subplots_plus_col = (subplots[0], subplots[1] + 1)
        self.fig, axs = plt.subplots(*subplots_plus_col)
        self.axs = np.array(axs).reshape(subplots_plus_col)  # type: ignore

        self.axs_anotate = self.axs[:, -1]
        self.axs = self.axs[0:, 0:-1]

        for ax in self.axs_anotate:
            ax.axis("off")
            ax.set_title(" ")

        self._remove_axs_anotate = True
        self._last_ax = (0, 0)
        self.metadata = ""

        self._axs_rescale = np.full(
            np.shape(self.axs),
            {
                "x_unit_prefix": None,
                "x_unit_scale": None,
                "y_unit_prefix": None,
                "y_unit_scale": None,
                "z_unit_prefix": None,
                "z_unit_scale": None,
            },
        )

    def set_default_settings(self, default_settings: Union[dict, str, bool] = True):
        """Sets the default settings for the plotter. This function is a wrapper for matplotlib.pyplot.style.use

            Args:
                default_settings (Union[dict, str, bool], optional): The default settings to use. Defaults to True.
            """
        if default_settings is True or default_settings == "quantum_calibrator":
            dirname = os.path.dirname(__file__)
            plt.style.use(
                os.path.join(dirname, r"plot_styles/quantum_calibrator.mplstyle")
            )

        elif default_settings == "presentation":
            dirname = os.path.dirname(__file__)
            plt.style.use(os.path.join(dirname, r"plot_styles/presentation.mplstyle"))

        elif default_settings is False:
            plt.style.use("default")

        elif isinstance(default_settings, dict):
            plt.rcParams.update(default_settings)

        elif isinstance(default_settings, str):
            plt.style.use(default_settings)

        else:
            raise ValueError(
                "default_settings must be either a dict, a string or a boolean"
            )

    @matplotlib_decorator
    def plot(
        self, x: Valueclass, y: Valueclass, ax: tuple = (), **kwargs,
    ):
        """plotting function for 1d data. This function is a wrapper for matplotlib.pyplot.plot

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is conv erted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """

        self.axs[self._last_ax].plot(x.value, y.value, **kwargs)

    @matplotlib_decorator
    def scatter(self, x: Valueclass, y: Valueclass, ax: tuple = (), **kwargs):
        """plotting function for 1d data. This function is a wrapper for matplotlib.pyplot.scatter

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        kwargs.setdefault("marker", "x")
        kwargs.setdefault("s", 30)

        self.axs[self._last_ax].scatter(
            x=x.value, y=y.value, **kwargs,
        )

    @matplotlib_decorator
    def bar(self, x: Valueclass, y: Valueclass, ax: tuple = (), **kwargs):
        """plotting function for 1d data. This function is a wrapper for matplotlib.pyplot.bar

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        kwargs.setdefault("width", 0.5)

        self.axs[self._last_ax].bar(x.value, y.value, **kwargs)

    @matplotlib_decorator
    def errorbar(self, x: Valueclass, y: Valueclass, ax: tuple = (), **kwargs):
        """plotting function for 1d data. This function is a wrapper for matplotlib.pyplot.errorbar

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        kwargs.setdefault("fmt", ".")
        kwargs.setdefault("elinewidth", 2)
        kwargs.setdefault("capsize", 3)

        self.axs[self._last_ax].errorbar(x.value, y.value, y.error, **kwargs)

    @matplotlib_decorator
    def _2d_genereal_plot(
        self,
        plot_type: str,
        x: Valueclass,
        y: Valueclass,
        z: Valueclass,
        ax: tuple = (),
        **kwargs,
    ):
        """general plotting function for 2d data. This function is a wrapper for matplotlib.pyplot

        Args:
            plot_type (str): The type of plot to use. This is the name of the function in matplotlib.pyplot. e.g. "contourf". Options: "contourf", "contour", "pcolormesh", "imshow".
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            z (Valueclass): z data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().

        Raises:
            ValueError: If plot_type is not a valid option.
        """
        kwargs.setdefault("cmap", "RdBu")
        kwargs.setdefault("vmin", np.min(z.value))
        kwargs.setdefault("vmax", np.max(z.value))

        if plot_type == "pcolormesh":
            c = self.axs[self._last_ax].pcolormesh(x.value, y.value, z.value, **kwargs)

        elif plot_type == "contour":
            c = self.axs[self._last_ax].contour(x.value, y.value, z.value, **kwargs)

        elif plot_type == "contourf":
            c = self.axs[self._last_ax].contourf(x.value, y.value, z.value, **kwargs)

        elif plot_type == "tricontour":
            c = self.axs[self._last_ax].tricontour(x.value, y.value, z.value, **kwargs)

        elif plot_type == "tricontourf":
            c = self.axs[self._last_ax].tricontourf(x.value, y.value, z.value, **kwargs)

        elif plot_type == "tripcolor":
            c = self.axs[self._last_ax].tripcolor(x.value, y.value, z.value, **kwargs)

        else:
            raise ValueError(f"plot_type {plot_type} not recognized")

        self.axs[self._last_ax].axis([x.value.min(), x.value.max(), y.value.min(), y.value.max()])  # type: ignore
        label = f"{z.name} [{z.unit}]" if z.unit else f"{z.name}"
        colorbar = self.fig.colorbar(c, ax=self.axs[self._last_ax], label=label)
        self.axs[self._last_ax].colorbar = colorbar

    def pcolormesh(
        self, x: Valueclass, y: Valueclass, Z: Valueclass, ax: tuple = (), **kwargs
    ):
        """plotting function for 2d data. This function is a wrapper for matplotlib.pyplot.pcolormesh

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            Z (Valueclass): z data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        self._2d_genereal_plot(plot_type="pcolormesh", x=x, y=y, z=Z, ax=ax, **kwargs)

    def heatmap(
        self, x: Valueclass, y: Valueclass, Z: Valueclass, ax: tuple = (), **kwargs
    ):
        """plotting function for 2d data. This function is a wrapper for matplotlib.pyplot.pcolormesh

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            Z (Valueclass): z data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        self._2d_genereal_plot(plot_type="pcolormesh", x=x, y=y, z=Z, ax=ax, **kwargs)

    def contour(
        self, x: Valueclass, y: Valueclass, Z: Valueclass, ax: tuple = (), **kwargs
    ):
        """plotting function for 2d data. This function is a wrapper for matplotlib.pyplot.contour

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            Z (Valueclass): z data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        self._2d_genereal_plot(plot_type="contour", x=x, y=y, z=Z, ax=ax, **kwargs)

    def contourf(
        self, x: Valueclass, y: Valueclass, Z: Valueclass, ax: tuple = (), **kwargs
    ):
        """plotting function for 2d data. This function is a wrapper for matplotlib.pyplot.contourf

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            Z (Valueclass): z data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        self._2d_genereal_plot(plot_type="contourf", x=x, y=y, z=Z, ax=ax, **kwargs)

    def tricontour(
        self, x: Valueclass, y: Valueclass, z: Valueclass, ax: tuple = (), **kwargs
    ):
        """plotting function for 2d data. This function is a wrapper for matplotlib.pyplot.tricontour

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            Z (Valueclass): z data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        self._2d_genereal_plot(plot_type="tricontour", x=x, y=y, z=z, ax=ax, **kwargs)

    def tricontourf(
        self, x: Valueclass, y: Valueclass, z: Valueclass, ax: tuple = (), **kwargs
    ):
        """plotting function for 2d data. This function is a wrapper for matplotlib.pyplot.tricontourf

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            Z (Valueclass): z data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        self._2d_genereal_plot(plot_type="tricontourf", x=x, y=y, z=z, ax=ax, **kwargs)

    def tripcolor(
        self, x: Valueclass, y: Valueclass, z: Valueclass, ax: tuple = (), **kwargs
    ):
        """plotting function for 2d data. This function is a wrapper for matplotlib.pyplot.tripcolor

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            Z (Valueclass): z data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        self._2d_genereal_plot(plot_type="tripcolor", x=x, y=y, z=z, ax=ax, **kwargs)

    def axhline(
        self,
        y: Union[Valueclass, tuple, list, float, np.ndarray],
        xmin: float = 0,
        xmax: float = 1,
        linestyle: str = "--",
        ax: tuple = (),
        **kwargs,
    ):
        """plotting function for 2d data. This function is a wrapper for matplotlib.pyplot.axhline

        Args:
            y (float): y value to plot.
            min (float): minimum x value to plot.
            max (float): maximum x value to plot.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        ax = ax or self._last_ax
        kwargs |= {"xmin": xmin, "xmax": xmax, "linestyle": linestyle}

        if isinstance(y, Valueclass):
            self.axs[self._last_ax].axhline(y=y.value, **kwargs)
        elif isinstance(y, (tuple, list)):
            for y_ in y:
                self.axs[self._last_ax].axhline(y=y_, **kwargs)
        else:
            self.axs[self._last_ax].axhline(y=y, **kwargs)

        [
            ax.legend()
            for ax in self.axs.flatten()
            if ax.get_legend_handles_labels() != ([], [])
        ]

    def axvline(
        self,
        x: Union[Valueclass, tuple, list, float, np.ndarray],
        ymin: float = 0,
        ymax: float = 1,
        linestyle: str = "--",
        ax: tuple = (),
        **kwargs,
    ):
        """plotting function for 2d data. This function is a wrapper for matplotlib.pyplot.axvline

        Args:
            x (float): x value to plot.
            min (float): minimum y value to plot.
            max (float): maximum y value to plot.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        ax = ax or self._last_ax
        kwargs |= {"ymin": ymin, "ymax": ymax, "linestyle": linestyle}

        if isinstance(x, Valueclass):
            self.axs[self._last_ax].axvline(x=x.value, **kwargs)
        elif isinstance(x, (tuple, list)):
            for x_ in x:
                self.axs[self._last_ax].axvline(x=x_, **kwargs)
        else:
            self.axs[self._last_ax].axvline(x=x, **kwargs)

        [
            ax.legend()
            for ax in self.axs.flatten()
            if ax.get_legend_handles_labels() != ([], [])
        ]

    @matplotlib_decorator
    def add_yresiuals(self, x: Valueclass, y: Valueclass, ax: tuple = (), **kwargs):
        """plotting function for 2d data. This function is a wrapper for matplotlib.pyplot.axvline

        Args:
            x (float): x value to plot.
            min (float): minimum y value to plot.
            max (float): maximum y value to plot.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        kwargs.setdefault("marker", "x")
        kwargs.setdefault("s", 30)

        ax = ax or self._last_ax
        ax_shape = np.shape(self.axs)
        axarg = np.where(self.axs.flatten() == self.axs[ax])[0][0]

        gs = gridspec.GridSpec(
            ax_shape[0],
            2 * ax_shape[1] + 2,
            width_ratios=[4, 1] * (ax_shape[1] + 1),
            wspace=0.00,
        )

        self.axs[ax].set_subplotspec(gs[axarg])
        self.yres = self.fig.add_subplot(gs[axarg + 1])

        self.yres.scatter(x.value, y.value, **kwargs)
        self.yres.axvline(x=0, linestyle=":", color="red")

        self.yres.sharey(self.axs[ax])
        self.yres.label_outer()  # type: ignore

        xlabel = kwargs.pop(
            "xlabel", f"Residuals [{x.unit}]" if x.unit else "Residuals"
        )
        self.yres.set_xlabel(xlabel)

    @matplotlib_decorator
    def add_xresiuals(self, x: Valueclass, y: Valueclass, ax: tuple = (), **kwargs):
        """plotting function for 2d data. This function is a wrapper for matplotlib.pyplot.axvline

        Args:
            x (float): x value to plot.
            min (float): minimum y value to plot.
            max (float): maximum y value to plot.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        kwargs.setdefault("marker", "x")
        kwargs.setdefault("s", 30)

        ax = ax or self._last_ax
        ax_shape = np.shape(self.axs)
        axarg = np.where(self.axs.flatten() == self.axs[ax])[0][0]

        gs = gridspec.GridSpec(
            2 * ax_shape[0],
            ax_shape[1] + 1,
            height_ratios=[4, 1] * (ax_shape[0]),
            hspace=0.00,
        )

        self.axs[ax].set_subplotspec(gs[axarg])
        self.xres = self.fig.add_subplot(gs[1 + ax_shape[0]])

        self.xres.scatter(x.value, y.value, **kwargs)
        self.xres.axhline(y=0, linestyle=":", color="red")

        # self.xres.sharex(self.axs[ax])
        self.axs[ax].sharex(self.xres)
        self.xres.set_xlabel(self.axs[ax].get_xlabel())
        # self.xres.label_outer()  # type: ignore
        self.axs[ax].label_outer()  # type: ignore

        ylabel = kwargs.pop(
            "ylabel", f"Residuals [{y.unit}]" if y.unit else "Residuals"
        )
        self.xres.set_ylabel(ylabel)

    def add_metadata(
        self, metadata: str, ax: tuple = (), overwrite: bool = False, **kwargs
    ):
        """Adds metadata to the plot. This is done by adding a text box to the plot.

        Args:
            metadata (dict): The metadata to add to the plot. This is a dictionary with the keys as the metadata name and the values as the metadata value.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
            overwrite (bool, optional): If True, the metadata is overwritten. If False, the metadata is added to the existing metadata. Defaults to False.
        """
        if ax:
            self._last_ax = ax

        self.metadata = metadata if overwrite else f"{self.metadata}{metadata}"
        ax_anotate = self.axs_anotate[self._last_ax[1]]

        x = kwargs.pop("x", 0.05)
        y = kwargs.pop("y", 0.95)
        va = kwargs.pop("va", "top")
        ha = kwargs.pop("ha", "left")
        transfrom = kwargs.pop("transform", ax_anotate.transAxes)

        ax_anotate.text(
            x, y, self.metadata, va=va, ha=ha, transform=transfrom, **kwargs
        )

        self._remove_axs_anotate = False

    def _get_default_transform(self):
        right_axs = self.axs[self._last_ax[0], np.size(self.axs, axis=1) - 1]
        if not hasattr(right_axs, "colorbar"):
            return self.axs[self._last_ax].transAxes
        print("colorbar:", right_axs.colorbar)
        return right_axs.colorbar.ax.transAxes

    def _label_with_unit_prefix(self, label: str, unit_prefix: str):
        return (
            label.replace("[", f"[{unit_prefix}")
            if "[" in label and "]" in label
            else f"{label} [{unit_prefix}]"
            if unit_prefix
            else label
        )

    def _rescale_axes(self):
        for ax in self.axs.flatten():
            xticks = ax.get_xticks() if hasattr(ax, "get_xticks") else None
            yticks = ax.get_yticks() if hasattr(ax, "get_yticks") else None
            cticks = ax.colorbar.get_ticks() if hasattr(ax, "colorbar") else None

            self._set_xticks(ax, xticks)
            self._set_yticks(ax, yticks)
            self._set_cticks(ax, cticks)

        if hasattr(self, "yres"):
            ax = self.yres
            xticks = ax.get_xticks() if hasattr(ax, "get_xticks") else None
            self._set_xticks(ax, xticks)

        if hasattr(self, "xres"):
            ax = self.xres
            xticks = ax.get_xticks() if hasattr(ax, "get_xticks") else None
            yticks = ax.get_yticks() if hasattr(ax, "get_yticks") else None
            self._set_xticks(ax, xticks)
            self._set_yticks(ax, yticks)

    def _set_xticks(self, ax, xticks):
        if xticks is not None:
            xlim = ax.get_xlim()
            ax.set_xticks(xticks)
            ax.set_xlim(xlim)
            rescaled_xticks, x_unit_prefix, _ = convert_array_with_unit(xticks)
            ax.set_xticklabels([f"{xtick:g}" for xtick in rescaled_xticks])
            if not hasattr(ax, "original_xlabel"):
                ax.original_xlabel = ax.get_xlabel()
            ax.set_xlabel(
                self._label_with_unit_prefix(ax.original_xlabel, x_unit_prefix)
            )

    def _set_cticks(self, ax, cticks):
        if cticks is not None:
            clim = ax.colorbar.ax.get_ylim()
            ax.colorbar.ax.set_yticks(cticks)
            ax.colorbar.ax.set_ylim(clim)
            rescaled_cticks, c_unit_prefix, _ = convert_array_with_unit(cticks)
            ax.colorbar.ax.set_yticklabels([f"{ctick:g}" for ctick in rescaled_cticks])
            if not hasattr(ax.colorbar, "original_label"):
                ax.colorbar.original_label = ax.colorbar.ax.get_ylabel()
            ax.colorbar.ax.set_ylabel(
                self._label_with_unit_prefix(ax.colorbar.original_label, c_unit_prefix)
            )

    def _set_yticks(self, ax, yticks):
        if yticks is not None:
            ylim = ax.get_ylim()
            ax.set_yticks(yticks)
            ax.set_ylim(ylim)
            rescaled_yticks, y_unit_prefix, _ = convert_array_with_unit(yticks)
            ax.set_yticklabels([f"{ytick:g}" for ytick in rescaled_yticks])
            if not hasattr(ax, "original_ylabel"):
                ax.original_ylabel = ax.get_ylabel()
            ax.set_ylabel(
                self._label_with_unit_prefix(ax.original_ylabel, y_unit_prefix)
            )

    def show(self):
        """Shows the plot. This function is a wrapper for matplotlib.pyplot.show

        Returns:
            fig: The figure object
        """
        if self._remove_axs_anotate:
            for ax in self.axs_anotate.flatten():
                with contextlib.suppress(ValueError):
                    ax.remove()

        self._rescale_axes()
        plt.figure(self.fig)
        plt.tight_layout()
        return plt.show()

    def save(self, path: str):
        """Saves the plot. This function is a wrapper for matplotlib.pyplot.savefig

        Args:
            path (str): The path to save the plot to.
        """
        plt.savefig(path)

