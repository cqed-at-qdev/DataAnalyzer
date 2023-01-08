# Author: Malthe Asmus Marciniak Nielsen
import os
from typing import Any, Union
from matplotlib import gridspec, ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from dataanalyzer.fitter import Fitter
from dataanalyzer.plotter.plotter_decorators import matplotlib_decorator
from dataanalyzer.utilities import (
    convert_array_with_unit,
    from_float_to_valueclass,
    Valueclass,
)


class Plotter:
    def __init__(
        self,
        subplots=(1, 1),
        default_settings: Union[dict, str, bool] = True,
        interactive: bool = False,
        **kwargs,
    ):
        """The Plotter class is a wrapper for matplotlib.pyplot. It is used to plot data in a consistent way.

        Args:
            subplots (tuple, optional): The shape of the subplots. Defaults to (1, 1).
        """

        if interactive:
            mpl.use("Qt5Agg")

        self.kwargs = kwargs

        self.set_default_settings(default_settings)
        self._setup_fig_and_ax(subplots)

        self.metadata = ""

    def _setup_fig_and_ax(self, subplots: tuple):
        self.fig: Figure = self.kwargs.pop("fig", None)
        subplots_plus_col = (subplots[0], subplots[1] + 1)

        if self.fig is None:
            self.fig, axs = plt.subplots(*subplots_plus_col)
        else:
            plt.close(self.fig)
            self.fig.clf()
            axs = self.fig.subplots(*subplots_plus_col)

        axs = np.array(axs).reshape(subplots_plus_col)
        self.axs: np.ndarray[Axes, Any] = axs[0:, -1]
        self.ax = self.axs[0, 0]

        self._setup_ax_anotate(ax_anotate=axs[0:, -1])

    def _setup_ax_anotate(self, ax_anotate: np.ndarray[Axes, Any]):
        gs = ax_anotate[0].get_gridspec()

        for ax in ax_anotate:
            ax.remove()

        self.ax_anotate = self.fig.add_subplot(gs[0:, -1])
        self.ax_anotate.axis("off")
        self._remove_ax_anotate = True

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

    def plot_fit(self, fit_obejct: Fitter, ax: tuple = (), **kwargs):
        """Plots a fit object. This function is a wrapper for matplotlib.pyplot.plot

        Args:
            fit_obejct (object): The fit object to plot.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        flip_axis = kwargs.pop("flip_axis", False)
        ls_start = kwargs.pop("linspace_start", None)
        ls_stop = kwargs.pop("linspace_stop", None)
        ls_steps = kwargs.pop("linspace_steps", 1000)

        if kwargs.pop("force_fit", False) or not fit_obejct._fitted:
            fit_obejct.do_fit(**kwargs)

        if kwargs.pop("plot_data", True):
            x, y = fit_obejct.x, fit_obejct.y

            if flip_axis:
                x, y = y, x

            self.scatter(x, y, ax=ax, label="Data")

        if kwargs.pop("plot_guess", True):
            x_guess, y_guess = fit_obejct.get_guess_array(ls_start, ls_stop, ls_steps)
            x_guess = from_float_to_valueclass(x_guess, "X guess")
            y_guess = from_float_to_valueclass(y_guess, "Y guess")

            if flip_axis:
                x_guess, y_guess = y_guess, x_guess

            self.plot(x_guess, y_guess, ax=ax, ls="--", color="grey", label="Guess")

        if kwargs.pop("plot_fit", True):
            x_fit, y_fit = fit_obejct.get_fit_array(ls_start, ls_stop, ls_steps)
            x_fit = from_float_to_valueclass(x_fit, "X fitted")
            y_fit = from_float_to_valueclass(y_fit, "Y fitted")

            if flip_axis:
                x_fit, y_fit = y_fit, x_fit

            self.plot(x_fit, y_fit, ax=ax, label="Fit")

        if kwargs.pop("plot_residuals", False):
            x_res = fit_obejct.x
            y_res = fit_obejct.get_residuals()

            if flip_axis:
                x_res, y_res = y_res, x_res
                self.add_yresiuals(x_res, y_res, ax=ax)
            else:
                self.add_xresiuals(x_res, y_res, ax=ax)

        if kwargs.pop("plot_metadata", True):
            self.add_metadata(fit_obejct._report_string, ax=ax)

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

        try:
            self.ax.plot(x.value, y.value, **kwargs)
        except ValueError:
            self.ax.plot(x.value, y.value.T, **kwargs)
            print("Warning: x and y have different shapes. Transposing y.")

    @matplotlib_decorator
    def scatter(self, x: Valueclass, y: Valueclass, ax: tuple = (), **kwargs):
        """plotting function for 1d data. This function is a wrapper for matplotlib.pyplot.scatter

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        # kwargs.setdefault("marker", "x")
        kwargs.setdefault("s", 30)

        try:
            self.ax.scatter(x.value, y.value, **kwargs)
        except ValueError:
            try:
                [
                    self.ax.scatter(x.value, y.value[i], **kwargs)
                    for i in range(y.value.shape[0])
                ]
            except ValueError:
                [
                    self.ax.scatter(x.value, y.value[:, i], **kwargs)
                    for i in range(y.value.shape[1])
                ]
                print("Warning: x and y have different shapes. Transposing y.")

    @matplotlib_decorator
    def bar(self, x: Valueclass, y: Valueclass, ax: tuple = (), **kwargs):
        """plotting function for 1d data. This function is a wrapper for matplotlib.pyplot.bar

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        kwargs.setdefault("width", 0.5)

        try:
            self.ax.bar(x.value, y.value, **kwargs)
        except ValueError:
            self.ax.bar(x.value, y.value.T, **kwargs)
            print("Warning: x and y have different shapes. Transposing y.")

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

        yerr = kwargs.pop("yerr", y.error)
        xerr = kwargs.pop("xerr", x.error)

        if isinstance(yerr, Valueclass):
            yerr = yerr.value
        if isinstance(xerr, Valueclass):
            xerr = xerr.value

        try:
            self.ax.errorbar(x=x.value, y=y.value, yerr=yerr, xerr=xerr, **kwargs)
        except ValueError:
            self.ax.errorbar(x=x.value, y=y.value.T, yerr=yerr.T, xerr=xerr, **kwargs)
            print("Warning: x and y have different shapes. Transposing y.")

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
        keep_colorbar = kwargs.get("keep_colorbar", False)
        kwargs.setdefault("cmap", "RdBu")
        kwargs.setdefault("vmin", np.min(z.value))
        kwargs.setdefault("vmax", np.max(z.value))

        if plot_type == "pcolormesh":
            c = self.ax.pcolormesh(x.value, y.value, z.value, **kwargs)

        elif plot_type == "contour":
            c = self.ax.contour(x.value, y.value, z.value, **kwargs)

        elif plot_type == "contourf":
            c = self.ax.contourf(x.value, y.value, z.value, **kwargs)

        elif plot_type == "tricontour":
            c = self.ax.tricontour(x.value, y.value, z.value, **kwargs)

        elif plot_type == "tricontourf":
            c = self.ax.tricontourf(x.value, y.value, z.value, **kwargs)

        elif plot_type == "tripcolor":
            c = self.ax.tripcolor(x.value, y.value, z.value, **kwargs)

        else:
            raise ValueError(f"plot_type {plot_type} not recognized")

        self.ax.axis([x.value.min(), x.value.max(), y.value.min(), y.value.max()])  # type: ignore

        # self._add_colorbar(c, z, keep_colorbar)

    def _add_colorbar(self, c, z, keep_colorbar):
        if hasattr(self.ax, "colorbar") and keep_colorbar:
            self.ax.colorbar.remove()

        label = f"{z.name} [{z.unit}]" if z.unit else f"{z.name}"
        colorbar = self.fig.colorbar(c, ax=self.ax, label=label)

        for ax in self.axs.flatten():
            if ax == self.ax:
                ax.colorbar = colorbar

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
        if ax:
            self.ax = self.axs[ax]

        kwargs |= {"xmin": xmin, "xmax": xmax, "linestyle": linestyle}

        if isinstance(y, Valueclass):
            self.ax.axhline(y=y.value, **kwargs)
        elif isinstance(y, (tuple, list)):
            for y_ in y:
                self.ax.axhline(y=y_, **kwargs)
        else:
            self.ax.axhline(y=y, **kwargs)

        self._plot_legends()

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
        if ax:
            self.ax = self.axs[ax]

        kwargs |= {"ymin": ymin, "ymax": ymax, "linestyle": linestyle}

        if isinstance(x, Valueclass):
            self.ax.axvline(x=x.value, **kwargs)
        elif isinstance(x, (tuple, list)):
            for x_ in x:
                self.ax.axvline(x=x_, **kwargs)
        else:
            self.ax.axvline(x=x, **kwargs)

        self._plot_legends()

    @matplotlib_decorator
    def add_yresiuals(self, x: Valueclass, y: Valueclass, ax: tuple = (), **kwargs):
        """plotting function for 2d data. This function is a wrapper for matplotlib.pyplot.axvline

        Args:
            x (float): x value to plot.
            min (float): minimum y value to plot.
            max (float): maximum y value to plot.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        # kwargs.setdefault("marker", "x")
        kwargs.setdefault("s", 30)

        ax_shape = np.shape(self.axs)
        axarg = np.where(self.axs.flatten() == self.ax)[0][0]

        gs = gridspec.GridSpec(
            ax_shape[0],
            2 * ax_shape[1] + 2,
            width_ratios=[4, 1] * (ax_shape[1] + 1),
            wspace=0.00,
        )

        self.ax.set_subplotspec(gs[axarg])
        self.yres = self.fig.add_subplot(gs[axarg + 1])

        self.yres.scatter(x.value, y.value, **kwargs)
        self.yres.axvline(x=0, linestyle=":", color="red")

        self.yres.sharey(self.ax)
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
        # kwargs.setdefault("marker", "x")
        kwargs.setdefault("s", 30)

        ax_shape = np.shape(self.axs)
        axarg = np.where(self.axs.flatten() == self.ax)[0][0]

        gs = gridspec.GridSpec(
            2 * ax_shape[0],
            ax_shape[1] + 1,
            height_ratios=[4, 1] * (ax_shape[0]),
            hspace=0.00,
        )

        self.ax.set_subplotspec(gs[axarg])
        self.xres = self.fig.add_subplot(gs[1 + ax_shape[0]])

        self.xres.scatter(x.value, y.value, **kwargs)
        self.xres.axhline(y=0, linestyle=":", color="red")

        if self.ax._sharex is None or self.xres is self.ax._sharex:
            self.ax.sharex(self.xres)
            self.xres.set_xlabel(self.ax.get_xlabel())
            self.ax.label_outer()  # type: ignore

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
            self.ax = self.axs[ax]

        self.metadata = metadata if overwrite else f"{self.metadata}{metadata}"
        x = kwargs.pop("x", 0.05)
        y = kwargs.pop("y", 0.95)
        va = kwargs.pop("va", "top")
        ha = kwargs.pop("ha", "left")
        transfrom = kwargs.pop("transform", self.ax_anotate.transAxes)

        self.ax_anotate.texts.clear()
        self.ax_anotate.text(
            x, y, self.metadata, va=va, ha=ha, transform=transfrom, **kwargs
        )

        self._remove_ax_anotate = False

    def clear_metadata(self):
        self.metadata = ""
        self.ax_anotate.texts.clear()

    def _get_default_transform(self):
        axarg = np.where(self.axs == self.ax)[0][0]
        right_axs = self.axs[axarg, np.size(self.axs, axis=1) - 1]

        return (
            right_axs.colorbar.ax.transAxes
            if hasattr(right_axs, "colorbar")
            else self.ax.transAxes
        )

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
            self._get_set_ticks("x")(self, ax)
            self._get_set_ticks("y")(self, ax)

            if hasattr(ax, "colorbar"):
                self._get_set_ticks("y")(self, ax.colorbar)
                self._get_set_ticks("x")(self, ax.colorbar)

        if hasattr(self, "yres"):
            self._get_set_ticks("x")(self, self.yres)

        if hasattr(self, "xres"):
            self._get_set_ticks("x")(self, self.xres)
            self._get_set_ticks("y")(self, self.xres)

    def _set_ticks(self, axis, ticks):
        _, unit_prefix, scale = convert_array_with_unit(ticks)
        ticks = ticker.FuncFormatter(lambda v, pos: "{0:g}".format(v * scale))
        axis.set_major_formatter(ticks)
        return unit_prefix

    def _get_set_ticks(self, axis: str = "x"):
        if axis not in ["x", "y"]:
            raise ValueError(f"Unknown axis: {axis}")

        orig_label = f"original_{axis}label"
        get_ticks = f"get_{axis}ticks"
        get_label = f"get_{axis}label"

        def _set_ticks(self, ax):
            if not hasattr(ax, get_ticks):
                return

            ticks = getattr(ax, get_ticks)()
            unit_prefix = self._set_ticks(getattr(ax, f"{axis}axis"), ticks)

            if not hasattr(ax, orig_label):
                setattr(ax, orig_label, getattr(ax, get_label)())

            original_label = getattr(ax, orig_label)
            updated_label = self._label_with_unit_prefix(original_label, unit_prefix)

            if axis == "x":
                ax.set_xlabel(updated_label)
            else:
                ax.set_ylabel(updated_label)

        return _set_ticks

    def _plot_legends(self):
        """Adds legends to the plot if the user has specified them."""
        [
            ax.legend()
            for ax in self.axs.flatten()
            if ax.get_legend_handles_labels() != ([], [])
        ]

    def _resize_figure(self, reverse=False):
        """Resizes the figure to fit the plot."""
        if self._remove_ax_anotate == reverse:
            return

        self._remove_ax_anotate = reverse
        # self._resize_figure_to_one_less_column(reverse)
        # self._even_spacing_in_columns(reverse)
        self._hide_axs_anotate(reverse)

    # def _resize_figure_to_one_less_column(self, reverse=False):
    #     n_cols = self.ax_anotate.get_subplotspec().get_geometry()[1]
    #     width, height = self.fig.get_size_inches()

    #     if reverse:
    #         self.fig.set_size_inches((width / (n_cols - 1)) * n_cols, height)
    #     else:
    #         self.fig.set_size_inches((width / n_cols) * (n_cols - 1), height)

    # def _even_spacing_in_columns(self, reverse=False):
    #     scaling = 1 if reverse else 0
    #     n_cols = self.ax_anotate.get_subplotspec().get_geometry()[1] - 1
    #     gridspec = self.ax_anotate.get_subplotspec().get_gridspec()

    #     if not reverse:
    #         gridspec.set_width_ratios([1] * n_cols + [scaling])

    def _hide_axs_anotate(self, reverse=False):
        self.ax_anotate.remove()
        # for ax in self.axs_anotate.flatten():
        #     ax.visible = reverse
        #     ax.remove()

    def show(self, return_fig: bool = False):
        """Shows the plot. This function is a wrapper for matplotlib.pyplot.show

        Returns:
            fig: The figure object
        """
        plt.figure(self.fig)

        if self._remove_ax_anotate:
            self._resize_figure()

        self._rescale_axes()
        plt.tight_layout()
        plt.pause(0.001)
        return self.fig if return_fig else plt.show()

    def save(self, path: str):
        """Saves the plot. This function is a wrapper for matplotlib.pyplot.savefig

        Args:
            path (str): The path to save the plot to.
        """
        plt.savefig(path)
