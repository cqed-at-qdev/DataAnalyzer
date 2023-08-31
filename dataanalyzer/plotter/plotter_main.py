# Author: Malthe Asmus Marciniak Nielsen
import os
from typing import Any, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib import gridspec, rc, ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dataanalyzer.fitter import Fitter
from dataanalyzer.plotter.plotter_decorators import matplotlib_decorator
from dataanalyzer.utilities import Valueclass, convert_array_with_unit


####################################################################################################
#                   Plotter Class                                                                  #
####################################################################################################
class Plotter:
    def __init__(
        self,
        subplots: Tuple[int, int] = (1, 1),
        default_settings: Union[dict, str, bool] = True,
        interactive: bool = False,
        **kwargs,
    ):
        """The Plotter class is a wrapper for matplotlib.pyplot. It is used to plot data in a consistent way.

        Args:
            subplots (tuple, optional): The shape of the subplots. Defaults to (1, 1).

        Raises:
            ValueError: If the default settings are not a valid type.
        """

        self.fig: Figure = kwargs.pop("fig", None)

        # Enable interactive mode
        if interactive:
            mpl.use("TkAgg") if os.name == "posix" else mpl.use("Qt5Agg")

        if kwargs.pop("set_style", False):
            self.set_mpl_style()

        self.kwargs = kwargs

        self.set_default_settings(default_settings)
        self._setup_fig_and_ax(subplots)

        self.metadata = ""

    ############# Main Functions ###################################################################
    def _setup_fig_and_ax(self, subplots: tuple) -> None:
        """Setup the figure and axes. If a figure is provided in the keyword arguments,
        it will be used. Otherwise, a new figure will be created.

        Args:
            subplots (tuple): The shape of the subplots.
        """
        # Set the number of columns to be one more than the number of subplots
        subplots_plus_col = (subplots[0], subplots[1] + 1)

        # If the figure doesn't exist, create it
        if self.fig is None:
            self.fig, axs = plt.subplots(*subplots_plus_col)
        else:
            self.fig.clf()
            axs = self.fig.subplots(*subplots_plus_col)

        # Reshape the axes into a grid
        axs = np.array(axs).reshape(subplots_plus_col)
        self.axs: np.ndarray[Axes, Any] = axs[0:, :-1]
        self.ax = self.axs[0, 0]

        # Set up the axis for the annotations
        self._setup_ax_anotate(ax_anotate=axs[0:, -1])

    def _setup_ax_anotate(self, ax_anotate: "np.ndarray[Axes, Any]"):
        """Setup the axis for the annotations. This function removes all axes from the figure and adds a new axis to the figure.

        Args:
            ax_anotate (np.ndarray[Axes, Any]): The axis to use for the annotations.
        """
        # Get the grid spec of the first axis
        gs = ax_anotate[0].get_gridspec()

        # Remove all axes from the figure
        for ax in ax_anotate:
            ax.remove()

        # Add the new axis to the figure
        self.ax_anotate = self.fig.add_subplot(gs[0:, -1])

        # Set the axis to invisible
        self.ax_anotate.axis("off")

        # Set a flag to remove the axis on the next update
        self._remove_ax_anotate = True

    @staticmethod
    def set_default_settings(default_settings: Union[dict, str, bool] = True) -> None:
        """Sets the default settings for the plotter. This function is a wrapper for matplotlib.pyplot.style.use

        Args:
            default_settings (Union[dict, str, bool], optional): The default settings to use. Defaults to True.
        """
        # If the user wants to use the default settings, load the default settings
        # from the quantum_calibrator.mplstyle file
        if default_settings == "quantum_calibrator":
            dirname = os.path.dirname(__file__)
            plt.style.use(
                os.path.join(dirname, r"plot_styles/quantum_calibrator.mplstyle")
            )

        # If the user wants to use the presentation settings, load the presentation
        # settings from the presentation.mplstyle file
        elif default_settings is True or default_settings == "presentation":
            dirname = os.path.dirname(__file__)
            plt.style.use(os.path.join(dirname, r"plot_styles/presentation.mplstyle"))

        # If the user wants to use no default settings, load the default matplotlib
        # settings
        elif default_settings is False:
            plt.style.use("default")

        # If the user passes a dictionary with settings, load these settings
        elif isinstance(default_settings, dict):
            plt.rcParams.update(default_settings)

        # If the user passes a string with a style, load this style
        elif isinstance(default_settings, str):
            plt.style.use(default_settings)

        # If the user passes anything else, raise an error
        else:
            raise ValueError(
                "default_settings must be either a dict, a string or a boolean"
            )

    @staticmethod
    def set_style(style) -> None:
        Plotter.set_default_settings(style)

    ############# 1D Plotting Functions ############################################################
    def plot_fit(
        self,
        fit_obejct: Fitter,
        ax: tuple = (),
        force_fit: bool = False,
        flip_axis: bool = False,
        plot_data: bool = True,
        plot_guess: bool = True,
        plot_fit: bool = True,
        plot_residuals: bool = True,
        plot_metadata: bool = True,
        **kwargs,
    ):
        """Plots a fit object. This function is a wrapper for matplotlib.pyplot.plot

        Args:
            fit_obejct (object): The fit object to plot.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """

        ls_start = kwargs.pop("linspace_start", None)
        ls_stop = kwargs.pop("linspace_stop", None)
        ls_steps = kwargs.pop("linspace_steps", 1000)

        fit_line_color = kwargs.pop("fit_line_color", "xkcd:dark grey")
        fit_line_width = kwargs.pop("fit_line_style", 1)

        if isinstance(ax, plt.Axes):
            self.ax = ax

        elif isinstance(ax, int):
            self.ax = self.axs.flatten()[ax]
        else:
            self.ax = self.axs[ax] if ax else self.ax

        if force_fit or not fit_obejct._fitted:
            fit_obejct.do_fit()

        if plot_data:
            x, y = fit_obejct.x_scaled, fit_obejct.y_scaled
            y.error = fit_obejct.yerr_scaled.value

            if flip_axis:
                x, y = y, x

            # self.scatter(x, y, ax=ax, label="Data", **kwargs.pop("kwargs_data", {}))
            self.errorbar(
                x,
                y,
                label="Data",
                **kwargs.pop("kwargs_data", {}),
                ls="",
                marker=".",
                # markerfacecolor="white",
                # # color="black",
                # linewidth=1.5,
                # markersize=8,
            )

        if plot_guess:
            x_guess, y_guess = fit_obejct.get_guess_array(ls_start, ls_stop, ls_steps)
            x_guess = Valueclass.fromfloat(x_guess, "X guess")
            y_guess = Valueclass.fromfloat(y_guess, "Y guess")

            if flip_axis:
                x_guess, y_guess = y_guess, x_guess

            self.plot(x_guess, y_guess, ls="--", color="grey", label="Guess")

        if plot_fit:
            x_fit, y_fit = fit_obejct.get_fit_array(ls_start, ls_stop, ls_steps)

            x_fit = Valueclass.fromfloat(x_fit, "X fitted")
            y_fit = Valueclass.fromfloat(y_fit, "Y fitted")

            if flip_axis:
                x_fit, y_fit = y_fit, x_fit

            self.plot(x_fit, y_fit, label="Fit", c=fit_line_color, lw=fit_line_width)

        if plot_residuals:
            x_res, y_res = fit_obejct.get_residuals()

            if flip_axis:
                x_res, y_res = y_res, x_res
                self.add_yresiuals(x_res, y_res)
            else:
                self.add_xresiuals(x_res, y_res)

        if plot_metadata:
            self.add_metadata(fit_obejct.get_report())  # fit_obejct._report_string

    @matplotlib_decorator
    def plot(self, x: Valueclass, y: Valueclass, ax: tuple = (), **kwargs):
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
    def scatter(
        self,
        x: Valueclass,
        y: Valueclass,
        z: Valueclass = None,
        ax: tuple = (),
        **kwargs,
    ):
        """plotting function for 1d data. This function is a wrapper for matplotlib.pyplot.scatter

        Args:
            x (Valueclass): x data to plot. This data is converted to a Valueclass object if it is not already one.
            y (Valueclass): y data to plot. This data is converted to a Valueclass object if it is not already one.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """
        # kwargs.setdefault("marker", "x")
        kwargs.setdefault("s", kwargs.pop("size", 20))
        kwargs.setdefault("facecolor", kwargs.pop("fc", "white"))

        if z is not None:
            kwargs.setdefault("c", z.value)
            kwargs.setdefault("cmap", "viridis")

        try:
            cax = self.ax.scatter(x.value, y.value, **kwargs)
            if z is not None:
                self._add_colorbar(cax, z, False)

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
        kwargs.setdefault("elinewidth", 2)
        kwargs.setdefault("capsize", 3)
        kwargs.setdefault("ls", "")
        kwargs.setdefault("marker", ".")
        kwargs.setdefault("markerfacecolor", kwargs.pop("mfc", "white"))
        kwargs.setdefault("markersize", kwargs.pop("ms", 7))

        yerr = kwargs.pop("yerr", y.error)
        xerr = kwargs.pop("xerr", x.error)

        if isinstance(yerr, Valueclass):
            yerr = yerr.value
        if isinstance(xerr, Valueclass):
            xerr = xerr.value

        # print("x Error: ----------------------------------")
        # print("xerr: ", xerr)
        # print("xerr.any(): ", xerr.any())
        # print("xerr.shape: ", xerr.shape)

        # print("\n")

        # print("y Error: ----------------------------------")
        # print("yerr: ", yerr)
        # print("yerr.any(): ", yerr.any())
        # print("yerr.shape: ", yerr.shape)

        if np.isnan(yerr).all():
            yerr = None

        if np.isnan(xerr).all():
            xerr = None

        self.ax.errorbar(x=x.value, y=y.value, yerr=yerr, xerr=xerr, **kwargs)

    @matplotlib_decorator
    def violinplot(self, x: Valueclass, y: Valueclass, ax: tuple = (), **kwargs):
        self.ax.violinplot(y.value, x.value, **kwargs)

    ############# 2D Plotting Functions ############################################################
    @matplotlib_decorator
    def _2d_genereal_plot(
        self,
        plot_type: str,
        x: Valueclass,
        y: Valueclass,
        z: Valueclass,
        ax: tuple = (),
        add_colorbar: bool = True,
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
        keep_colorbar = kwargs.pop("keep_colorbar", False)
        kwargs.setdefault("cmap", "RdYlBu_r")
        kwargs.setdefault("vmin", np.nanmin(z.value))
        kwargs.setdefault("vmax", np.nanmax(z.value))

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

        if add_colorbar:
            self._add_colorbar(c, z, keep_colorbar)

    def _add_colorbar(self, c, z, keep_colorbar):
        if hasattr(self.ax, "colorbar") and not keep_colorbar:
            for colorbar in self.ax.colorbar:
                colorbar.remove()

        label = f"{z.name} [{z.unit}]" if z.unit else f"{z.name}"

        if c is None:
            # Make a dummy plot with min and max values to get the colorbar
            axdummy = self.fig.add_subplot(111)
            c = axdummy.scatter(x=[0, 1], y=[0, 1], c=[z.value.min(), z.value.max()])
            axdummy.remove()

        cbar_ax = self.ax.inset_axes(
            [0.7, 1.05, 0.3, 0.05], transform=self.ax.transAxes
        )
        colorbar = self.fig.colorbar(
            c, ax=self.ax, cax=cbar_ax, orientation="horizontal"
        )
        colorbar.ax.xaxis.set_ticks_position("top")
        colorbar.ax.set_ylabel(label)
        colorbar.ax.yaxis.label.set(rotation="horizontal", ha="right", va="center")

        for ax in self.axs.flatten():
            if ax == self.ax:
                if not hasattr(ax, "colorbar"):
                    ax.colorbar = [colorbar]
                else:
                    ax.colorbar.append(colorbar)

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

    ############# Other Plotting Functions #########################################################
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

        if "cycle_color" in kwargs:
            mpl_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            kwargs["color"] = mpl_cycle[kwargs.pop("cycle_color") % len(mpl_cycle)]

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

        if "cycle_color" in kwargs:
            mpl_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            kwargs["color"] = mpl_cycle[kwargs.pop("cycle_color") % len(mpl_cycle)]

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
        kwargs.setdefault("s", 15)

        if not hasattr(self.ax, "yres"):
            divider = make_axes_locatable(self.ax)
            self.ax.yres = divider.append_axes("right", size="20%", pad=0)
            self.ax.yres.axvline(x=0, linestyle=":", color="red")
            self.ax.yres.sharey(self.ax)

            xlabel = kwargs.pop(
                "xlabel", f"Residuals [{x.unit}]" if x.unit else "Residuals"
            )
            self.ax.yres.set_xlabel(xlabel)

        self.ax.yres.scatter(x.value, y.value, **kwargs)
        self.ax.yres.yaxis.set_tick_params(labelleft=False)  # hide the yticklabels

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
        kwargs.setdefault("s", 15)

        if not hasattr(self.ax, "xres"):
            divider = make_axes_locatable(self.ax)
            self.ax.xres = divider.append_axes("bottom", size="20%", pad=0)
            self.ax.xres.axhline(y=0, linestyle=":", color="red")

            if self.ax._sharex is None or self.ax.xres is self.ax._sharex:
                self.ax.sharex(self.ax.xres)
                self.ax.xres.set_xlabel(self.ax.get_xlabel())
                self.ax.label_outer()  # type: ignore

                ylabel = kwargs.pop(
                    "ylabel", f"Residuals [{y.unit}]" if y.unit else "Residuals"
                )
                self.ax.xres.set_ylabel(ylabel)

        self.ax.xres.scatter(x.value, y.value, **kwargs)

    def devide_ax(
        self,
        ax: tuple = (),
        position="bottom",
        size="100%",
        pad=0,
        share_ax=True,
        **kwargs,
    ):
        """Devide the ax into subplots."""

        if isinstance(ax, plt.Axes):
            self.ax = ax

        elif isinstance(ax, int):
            self.ax = self.axs.flatten()[ax]
        else:
            self.ax = self.axs[ax] if ax else self.ax

        divider = make_axes_locatable(self.ax)
        self.ax.subax = divider.append_axes(position, size=size, pad=pad, **kwargs)

        if share_ax:
            if position == "bottom":
                self.ax.sharex(self.ax.subax)
                self.ax.subax.set_xlabel(self.ax.get_xlabel())
                self.ax.set_xlabel("")
                self.ax.xaxis.set_tick_params(labelbottom=False)  # hide the xticklabels

            elif position == "left":
                self.ax.sharey(self.ax.subax)
                self.ax.subax.set_ylabel(self.ax.get_ylabel())
                self.ax.label_outer()

    ############# Metadata Functions ###############################################################
    def add_metadata(
        self,
        *metadata: Union[
            str, Valueclass, list[Valueclass], tuple[Valueclass], dict[str, Valueclass]
        ],
        ax: tuple = (),
        overwrite: bool = False,
        fontsize: int = 12,
        max_numb_of_lines: int = 30,
        **kwargs,
    ):
        """Adds metadata to the plot. This is done by adding a text box to the plot.

        Args:
            metadata (dict): The metadata to add to the plot. This is a dictionary with the keys as the metadata name and the values as the metadata value.
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
            overwrite (bool, optional): If True, the metadata is overwritten. If False, the metadata is added to the existing metadata. Defaults to False.
        """
        if ax:
            self.ax = self.axs[ax]

        default_kwargs = {
            "x": 0.05,
            "y": 0.95,
            "va": "top",
            "ha": "left",
            "transform": self.ax_anotate.transAxes,
            "fontdict": {"family": "monospace"},
        }

        kwargs = default_kwargs | kwargs  # type: ignore
        algin = kwargs.pop("algin", True)

        kwargs_metadata = {
            k.removeprefix("tostr_"): kwargs.pop(k)
            for k in list(kwargs)
            if k.startswith("tostr_")
        }
        metadata_str = self._convert_metadata_to_str(
            *metadata, algin=algin, **kwargs_metadata
        )
        self.metadata = metadata_str if overwrite else f"{self.metadata}{metadata_str}"

        # Check if the metadata is to long
        relative_size = self.metadata.count("\n") / max_numb_of_lines
        if relative_size > 1:
            fontsize /= relative_size

        self.ax_anotate.texts.clear()
        self.ax_anotate.text(s=self.metadata, fontsize=fontsize, **kwargs)
        self._remove_ax_anotate = False

    def _convert_metadata_to_str(
        self,
        *metadata: Union[
            str, Valueclass, list[Valueclass], tuple[Valueclass], dict[str, Valueclass]
        ],
        algin: bool = True,
        add_parameter_header: bool = True,
        **kwargs,
    ) -> str:
        name_width = kwargs.pop("tostr_name_width", 25)
        size_width = kwargs.pop("tostr_size_width", 6)

        # parameter_header = f"{Mesuerment parameters : <{name_width}}" TODO: add parameter header
        # {"N points" : <{size_width}}Values"

        metadata_str = ""
        for param in metadata:
            if isinstance(param, str):
                metadata_str += f"{param}\n"

            elif isinstance(param, Valueclass):
                metadata_str += f"{param.tostr(algin=algin, name_width=name_width, size_width=size_width, scale_values=param.has_unit, **kwargs)}\n"

            elif isinstance(param, (list, tuple)):
                for par in param:
                    metadata_str += self._convert_metadata_to_str(par)

            elif isinstance(param, dict):
                for par in param.values():
                    metadata_str += self._convert_metadata_to_str(par)

            else:
                raise TypeError(
                    f"metadata must be of type str, Valueclass, list[Value], tuple[Value], or dict[str, Value], not {type(param)}"
                )

        return metadata_str

    def clear_metadata(self):
        self.metadata = ""
        self.ax_anotate.texts.clear()

    ############# Quantum Plotting Functions #######################################################
    def add_3D_plot(self, ax: tuple = (), **kwargs):
        if isinstance(ax, plt.Axes):
            self.ax = ax

        elif isinstance(ax, int):
            self.ax = self.axs.flatten()[ax]
        else:
            self.ax = self.axs[ax] if ax else self.ax

        ax_idx = np.where(self.axs.flatten() == self.ax)[0][0]
        ax_insert = np.where(self.axs == self.ax)
        ax_insert = (ax_insert[0][0], ax_insert[1][0])

        row, col = self.axs.shape

        ax_idx += ax_idx // col

        # Convert ax to 3D if it is 2D
        ax_temp = self.ax
        self.ax = self.fig.add_subplot(row, col + 1, ax_idx + 1, projection="3d")
        ax_temp.remove()

        # Insert ax back into axs
        self.axs[ax_insert] = self.ax

    def add_bloch_sphere(self, ax: tuple = (), **kwargs):
        """Adds a bloch sphere to the plot.

        Args:
            ax (tuple, optional): The ax to use. If None, self._last_ax is used. Defaults to ().
        """

        font_size = kwargs.pop("font_size", 14)
        point_size = kwargs.pop(
            "point_size", [55, 62, 65, 75]
        )  # [7,7,7,7] for many points
        view = kwargs.pop("view", [-60, 30])

        if isinstance(ax, plt.Axes):
            self.ax = ax

        elif isinstance(ax, int):
            self.ax = self.axs.flatten()[ax]
        else:
            self.ax = self.axs[ax] if ax else self.ax

        ax_idx = np.where(self.axs.flatten() == self.ax)[0][0]
        ax_insert = np.where(self.axs == self.ax)
        ax_insert = (ax_insert[0][0], ax_insert[1][0])

        row, col = self.axs.shape

        ax_idx_temp = ax_idx + ax_idx // col

        # Convert ax to 3D if it is 2D
        ax_temp = self.ax
        self.ax = self.fig.add_subplot(row, col + 1, ax_idx_temp + 1, projection="3d")
        ax_temp.remove()

        # Insert ax back into axs
        self.axs[ax_insert] = self.ax

        # Add bloch sphere (if it doesn't exist)
        if not hasattr(self.ax, "bloch"):
            self.ax.bloch = qt.Bloch(fig=self.fig, axes=self.ax, **kwargs)
            self.ax.bloch.font_size = font_size
            self.ax.bloch.point_size = point_size
            # self.ax.bloch.view = view
            self.ax.bloch.render()

    def add_bloch_vector(self, vector: np.ndarray, ax: tuple = (), **kwargs):
        if isinstance(ax, plt.Axes):
            self.ax = ax

        elif isinstance(ax, int):
            self.ax = self.axs.flatten()[ax]
        else:
            self.ax = self.axs[ax] if ax else self.ax

        view = kwargs.pop("view", [-60, 30])

        # Check if bloch sphere exists
        if not hasattr(self.ax, "bloch"):
            self.add_bloch_sphere(ax=ax, view=view)

        # Add vector
        self.ax.bloch.add_states(vector, **kwargs)
        self.ax.bloch.render()

    def add_bloch_point(self, vector: np.ndarray, z=None, ax: tuple = (), **kwargs):
        keep_colorbar = kwargs.pop("keep_colorbar", False)
        cmap = kwargs.pop("cmap", "viridis")
        font_size = kwargs.pop("font_size", 14)
        point_size = kwargs.pop(
            "point_size", [55, 62, 65, 75] if np.shape(vector)[1] < 10 else [7, 7, 7, 7]
        )  # [7,7,7,7] for many points
        point_color = kwargs.pop("point_color", None)

        if isinstance(ax, plt.Axes):
            self.ax = ax

        elif isinstance(ax, int):
            self.ax = self.axs.flatten()[ax]
        else:
            self.ax = self.axs[ax] if ax else self.ax

        view = kwargs.pop("view", [-60, 30])

        # Check if bloch sphere exists
        if not hasattr(self.ax, "bloch"):
            self.add_bloch_sphere(
                ax=ax, font_size=font_size, point_size=point_size, view=view
            )

        if point_color is not None:
            if len(np.array(point_color)) == 1:
                point_color = [point_color for _ in range(len(vector))]
            self.ax.bloch.point_color = point_color

            [point_color for _ in range(len(vector))]
            kwargs.update({"meth": "m"})

        # Change color of point if z is not None
        elif z is not None:
            # Create dommy plot to get c
            c = plt.imshow(np.array([[z.min(), z.max()]]), cmap=cmap)
            c.set_visible(False)

            # Get list of colors
            colors = c.to_rgba(z.value)

            self.ax.bloch.point_color = colors
            kwargs.update({"meth": "m"})

        # Add vector
        self.ax.bloch.add_points(vector, **kwargs)
        self.ax.bloch.render()

        # Add colorbar if c is not None
        if point_color is None and z is not None:
            self._add_colorbar(c=c, z=z, keep_colorbar=keep_colorbar)

    ############# Other Functions ##################################################################
    def clear_ax(
        self,
        ax: plt.Axes = None,
        clear_title: bool = True,
        clear_xlabel: bool = True,
        clear_xticklabels: bool = True,
        clear_ylabel: bool = True,
        clear_yticklabels: bool = True,
        clear_colorbar: bool = True,
    ) -> None:
        if ax is None:
            ax = self.ax

        if clear_title:
            ax.set_title("")
        if clear_xlabel:
            ax.set_xlabel("")
        if clear_xticklabels:
            ax.set_xticklabels([])
        if clear_ylabel:
            ax.set_ylabel("")
        if clear_yticklabels:
            ax.set_yticklabels([])
        if clear_colorbar:
            ax.colorbar[0].remove()

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
                for colorbar in ax.colorbar:
                    self._get_set_ticks("y", "x")(self, colorbar)
                    self._get_set_ticks("x", "y")(self, colorbar)

            if hasattr(ax, "xres"):
                self._get_set_ticks("x")(self, ax.xres)
                self._get_set_ticks("y")(self, ax.xres)

            if hasattr(ax, "yres"):
                self._get_set_ticks("x")(self, ax.yres)
                self._get_set_ticks("y")(self, ax.yres)

            if hasattr(ax, "subax"):
                self._get_set_ticks("x")(self, ax.subax)
                self._get_set_ticks("y")(self, ax.subax)

    def _set_ticks(self, axis, ticks):
        _, unit_prefix, scale = convert_array_with_unit(ticks)
        ticks = ticker.FuncFormatter(lambda v, pos: "{0:g}".format(v * scale))
        axis.set_major_formatter(ticks)
        return unit_prefix

    def _get_set_ticks(self, axis: str = "x", label_axis=None):
        if axis not in ["x", "y"]:
            raise ValueError(f"Unknown axis: {axis}")

        if label_axis is None:
            label_axis = axis

        orig_label = f"original_{label_axis}label"
        get_ticks = f"get_{axis}ticks"
        get_label = f"get_{label_axis}label"

        def _set_ticks(self, ax):
            if not hasattr(ax, get_ticks):
                ax = ax.ax

            original_label = _get_original_label(ax)

            if any(x not in original_label for x in ["[", "]"]):
                return

            ticks = getattr(ax, get_ticks)()
            unit_prefix = self._set_ticks(getattr(ax, f"{axis}axis"), ticks)

            updated_label = self._label_with_unit_prefix(original_label, unit_prefix)

            if label_axis == "x":
                ax.set_xlabel(updated_label)
            else:
                ax.set_ylabel(updated_label)

            if label_axis != axis:
                ax.yaxis.label.set(rotation="horizontal", ha="right", va="center")

        def _get_original_label(ax):
            if not hasattr(ax, orig_label):
                setattr(ax, orig_label, getattr(ax, get_label)())

            original_label = getattr(ax, orig_label)
            return original_label

        return _set_ticks

    def _plot_legends(self):
        """Adds legends to the plot if the user has specified them."""
        [
            ax.legend()  # bbox_to_anchor=(1.04, 1), loc="upper left"
            for ax in self.axs.flatten()
            if ax.get_legend_handles_labels() != ([], []) and not hasattr(ax, "bloch")
        ]
        [
            ax.subax.legend()  # bbox_to_anchor=(1.04, 1), loc="upper left"
            for ax in self.axs.flatten()
            if hasattr(ax, "subax") and ax.subax.get_legend_handles_labels() != ([], [])
        ]

    def _make_axes_bold(self):
        """Makes all axes labels bold."""
        if not self.kwargs.get("bold_labels", True):
            return

        for ax in self.axs.flatten():
            ax.xaxis.label.set_fontweight("bold")
            ax.yaxis.label.set_fontweight("bold")

            if hasattr(ax, "colorbar"):
                for colorbar in ax.colorbar:
                    colorbar.ax.yaxis.label.set_fontweight("bold")

            if hasattr(ax, "xres"):
                ax.xres.xaxis.label.set_fontweight("bold")
                ax.xres.yaxis.label.set_fontweight("bold")

            if hasattr(ax, "yres"):
                ax.yres.xaxis.label.set_fontweight("bold")

            if hasattr(ax, "subax"):
                ax.subax.xaxis.label.set_fontweight("bold")
                ax.subax.yaxis.label.set_fontweight("bold")

    def _final_formatting(self):
        """Performs final formatting on the plot."""
        plt.figure(self.fig)

        if self._remove_ax_anotate:
            self._remove_ax_anotate = False
            self.ax_anotate.remove()

        self._plot_legends()
        self._make_axes_bold()
        self._rescale_axes()

        plt.tight_layout()

    def show(self, return_fig: bool = False):
        """Shows the plot. This function is a wrapper for matplotlib.pyplot.show

        Returns:
            fig: The figure object
        """

        self._final_formatting()  # TODO: How to use custom tick labels e.g. ["0", "π/2", "π"]?

        if not return_fig:
            return plt.show()

        for fig_num in plt.get_fignums():
            if self.fig.number != fig_num:  # type: ignore
                plt.close(fig_num)

        plt.pause(0.001)
        return self.fig

    def save(self, path: str, **kwargs):
        """Saves the plot. This function is a wrapper for matplotlib.pyplot.savefig

        Args:
            path (str): The path to save the plot to.
        """
        self._final_formatting()
        plt.savefig(path, **kwargs)
