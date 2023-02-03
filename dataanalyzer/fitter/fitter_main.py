# Author: Malthe Asmus Marciniak Nielsen
import copy
from distutils.spawn import find_executable
from typing import Union

import numpy as np
from iminuit import Minuit
from scipy import stats

from dataanalyzer.fitter import ExternalFunctions, Fitparam
from dataanalyzer.fitter.models.fitter_models import ModelABC
from dataanalyzer.utilities import Valueclass
from dataanalyzer.utilities.utilities import convert_array_with_unit, round_on_error


####################################################################################################
#                   Fitter Class                                                                   #
####################################################################################################
class Fitter:
    def __init__(
        self,
        func: ModelABC,
        x: Union[Valueclass, list, tuple, np.ndarray],
        y: Union[Valueclass, list, tuple, np.ndarray],
        sy=None,
        cost_function="chi2_no_uncertainty",
        use_latex: bool = False,
        **kwargs,
    ) -> None:
        """The fitter class

        Args:
            func (ModelABC): The model to be fitted
            x (Union[Valueclass, list, tuple, np.ndarray]): The x values. Can be a Valueclass object or a list, tuple or numpy array
            y (Union[Valueclass, list, tuple, np.ndarray]): The y values. Can be a Valueclass object or a list, tuple or numpy array
            sy (np.ndarray, optional): The uncertainty of the y values. Defaults to None.
            cost_function (str, optional): The cost function to be used in the fit. Defaults to "chi2".
            use_latex (bool, optional): If the report should be in latex format. Defaults to True.

        Raises:
            ValueError: If the x and y values are not in the correct format
        """
        self.func = func
        self.kwargs = kwargs
        self.param_names = copy.deepcopy(func._full_name_list)

        self._convert_and_scale_xy_data(x, y, sy)

        self.set_initial_values(self.kwargs.pop("intial_values", {}))
        self._set_initial_fitting_options(cost_function, use_latex)

    ############# Main Functions ###################################################################
    def _convert_data(self, x, y):
        for data_name, data in zip(["x", "y"], [x, y]):
            if not data:
                raise ValueError(f"{data_name} data is None")

            setattr(self, data_name, data)
            setattr(self, f"_{data_name}", Valueclass.fromfloat(copy.copy(data), f"{data_name} data"))

    def _scale_data(self):
        for data_name in ["x", "y"]:
            scaled_data, unit_prefix, conversion_factor = convert_array_with_unit(getattr(self, f"_{data_name}").value)
            getattr(self, f"_{data_name}").value = scaled_data
            setattr(self, f"_{data_name}_unit_prefix", unit_prefix)
            setattr(self, f"_{data_name}_cf", conversion_factor)

    def _convert_and_scale_xy_data(self, x, y, sy=None):
        self._convert_data(x, y)
        self._ftt_xy_data()
        self._scale_data()

        self._sy = sy * self._y_cf if sy is not None else None
        self._conversion_factors = self.func.get_units(x=self._x_cf, y=self._y_cf)

    def _ftt_xy_data(self):
        if self._y.fft_type == "fft_y" and self._x.fft_type != "fft_x":
            self._x = copy.deepcopy(self._x.fftfreq)
        elif self._y.fft_type == "fft_x" and self._x.fft_type != "fft_y":
            self._x = copy.deepcopy(self._x.fft)
        elif self._x.fft_type == "fft_x" and self._y.fft_type != "fft_x":
            self._y = copy.deepcopy(self._y.fftfreq)
        elif self._x.fft_type == "fft_y" and self._y.fft_type != "fft_y":
            self._y = copy.deepcopy(self._y.fft)

    @property
    def initial_guess(self):
        """Returns the initial guess of the fit parameters

        Returns:
            dict: Dictionary with the initial guess of the fit parameters
        """
        return {p: self.intial_values[p].values for p in self.param_names}

    def do_fit(
        self,
        linspace_start=None,
        linspace_stop=None,
        linspace_steps=1000,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, dict, str]:
        """Do the fit

        Args:
            return_linspace (bool, optional): If True a linspace of evalated points from linspace_start to linspace_stop of linspace_steps points is return. Defaults to True.
            linspace_start (float, optional): The start of the linspace. Defaults to None. Linestart is set to the minimum of the x data if None
            linspace_stop (float, optional): The stop of the linspace. Defaults to None. Linestop is set to the maximum of the x data if None
            linspace_steps (int, optional): The number of points in the linspace. Defaults to 1000.

        Returns:
            Union[tuple[np.ndarray, np.ndarray, str], str]: If return_linspace is True a tuple with the linspace, the evaluated function and the latex string is returned. If return_linspace is False only the latex string is returned
        """
        self._do_fit()

        return_list = []

        if kwargs.pop("return_fit_array", True):
            return_list += self.get_fit_array(linspace_start, linspace_stop, linspace_steps)

        if kwargs.pop("return_params", True):
            return_list.append(self.get_params())

        if kwargs.pop("return_report", True):
            return_list.append(self._report_string)

        if kwargs.pop("return_guess_array", False):
            return_list.append(self.get_guess_array(linspace_start, linspace_stop, linspace_steps))

        return tuple(return_list)

    def _do_fit(self):
        self.set_minuit()
        self.minuit.migrad()
        self._calculate_prob_and_make_report()
        self.params = self.get_params()
        self._fitted = True

    def _convert_x_and_y_to_valueclass(self, x_array, y_array):
        if isinstance(self.x, Valueclass):
            x_array = Valueclass(value=x_array, name=self.x.name, unit=self.x.unit)
        if isinstance(self.y, Valueclass):
            y_array = Valueclass(value=y_array, name=self.y.name, unit=self.y.unit)
        return x_array / self._x_cf, y_array / self._y_cf

    def _calculate_prob_and_make_report(self):
        """Calculate the probability of the fit"""
        self.nvar = len(self.minuit.parameters)  # Number of variables
        self.ndof = len(self._x.value) - self.nvar  # type: ignore # Ndof = n data points - n variables
        self.chi2 = self.minuit.fval  # The chi2 value
        self.prob = stats.chi2.sf(self.chi2, self.ndof)  # The chi2 probability
        self._update_iminuit_report()

    def _update_iminuit_report(self):
        """Update the iminuit report"""
        self._update_iminuit_report_with_function()
        self._update_imuniut_report_with_parameters()
        self._update_iminuit_report_with_statistics()

    def _update_iminuit_report_with_function(self):
        """Update the iminuit report with the function"""
        funcname_str = f"Function: \n{self.func.funcname()}\n\n"
        if len(funcname_str) > 100:
            funcname_str = f"Function: \n{self.func.funcname()[:75]}...\n\n"
        self._report_string = funcname_str

    def _update_imuniut_report_with_parameters(self):
        """Update the iminuit report with the parameters"""
        poly_degree = getattr(self.func, "poly_degree", None)
        values = self.minuit.values.to_dict().items()
        errors = self.minuit.errors.to_dict().values()

        unit_names = self._get_func_units()

        self._report_string += "Fit parameters:\n"

        for i, ((key, val), err) in enumerate(zip(values, errors)):
            if key in self.param_names:
                unit = f"$\mathrm{{{unit_names[key]}}}$" if unit_names else ""
                name = f"${self.func._display_name_list[i]}$"
                self._add_parameter_string(unit, name, val, err)

            if poly_degree is not None and i > poly_degree - 1:
                break

    def _add_parameter_string(self, unit, key, val, err):
        value_string = round_on_error(val, err, n_digits=2)
        self._report_string += f"{key}: {value_string} {unit}\n"

    def _update_iminuit_report_with_statistics(self):
        """Update the iminuit report with the probability and the chi2 value"""
        if any([self.chi2, self.ndof, self.prob]):
            self._report_string += "\nFit statistics:\n"

        if self.chi2 is not None:
            self._report_string += f"Chi²: {np.format_float_scientific(self.chi2, precision=4)}\n"

        if self.ndof is not None:
            self._report_string += f"Degree of freedom: {self.ndof}\n"

        if self.prob is not None:
            self._report_string += f"Probability: {self.prob}\n"

        self._report_string += "\n"

    ############# Get Methods ######################################################################
    def get_residuals(self):
        residuals = (self._y.value - self.func.func(x=self._x.value, **self.minuit.values.to_dict())) / self._y_cf
        return Valueclass(value=residuals, name=self._y.name, unit=self._y.unit)

    def get_fit_array(self, linspace_start=None, linspace_stop=None, linspace_steps=1000):
        x_fit = self._get_linspace_of_x(linspace_start, linspace_stop, linspace_steps)
        y_fit = np.array(self.func.func(x=x_fit, **self.minuit.values.to_dict()))

        x_fit, y_fit = self._convert_x_and_y_to_valueclass(x_fit, y_fit)
        return x_fit, y_fit

    def get_guess_array(self, linspace_start=None, linspace_stop=None, linspace_steps=1000):
        x_guess = self._get_linspace_of_x(linspace_start, linspace_stop, linspace_steps)
        y_guess = np.array(self.func.func(x=x_guess, **self.initial_guess))

        x_guess, y_guess = self._convert_x_and_y_to_valueclass(x_guess, y_guess)
        return x_guess, y_guess

    def _get_linspace_of_x(self, linspace_start=None, linspace_stop=None, linspace_steps=1000):
        linspace_start = linspace_start or np.min(self._x.value)
        linspace_stop = linspace_stop or np.max(self._x.value)

        return np.linspace(linspace_start, linspace_stop, linspace_steps)

    def get_report(self):
        """Get the report string"""
        return self.minuit

    def get_params(self, *param_names) -> dict:
        """Get the parameters

        Returns:
            dict: A dictionary with the parameters
        """
        if not param_names:
            param_names = self.param_names

        return {p: self._get_converted_param(p) for p in param_names}

    def _get_converted_param(self, param: str):
        if param not in self._conversion_factors:
            return {
                "value": self.minuit.values[param],
                "error": self.minuit.errors[param],
            }

        convertion_fator = self._conversion_factors[param]
        value = self.minuit.values[param]
        error = self.minuit.errors[param]

        return {"value": value / convertion_fator, "error": error / convertion_fator}

    def get_extrema(self):
        if not hasattr(self, "params"):
            self.do_fit()

        return self.func.get_extrema(self.params)

    def get_period(self):
        if not hasattr(self, "params"):
            self.do_fit()

        return self.func.get_period(self.params)

    def get_initial_values(self):
        """Returns the initial values of the fit parameters

        Returns:
            dict: Dictionary with the initial values of the fit parameters
        """
        for p in self.param_names:
            self.intial_values[p].values = self.minuit.values[p]
            self.intial_values[p].limits = self.minuit.limits[p]
            self.intial_values[p].fixed = self.minuit.fixed[p]
        return self.intial_values

    def _get_func_units(self) -> dict:
        if not hasattr(self.func, "units"):
            return {}

        if self._x.unit is None or self._y.unit is None:
            return {}

        x_unit = self._x_unit_prefix + self._x.unit
        y_unit = self._y_unit_prefix + self._y.unit

        return self.func.get_units(x=x_unit, y=y_unit)  # type: ignore

    ############# Set Methods ######################################################################
    def set_symbols(self, **symbols: str):
        self.func.set_symbols(**symbols)

    def set_units(self, **units: Union[str, float, int]):
        self.func.set_units(**units)

    def set_parameters(self, **values: Union[Fitparam, dict, float, int]):
        self.func.set_parameters(**values)

    def _set_initial_fitting_options(self, cost_function, use_latex):
        self.weights = self.kwargs.pop("weights", None)
        self.bound = self.kwargs.pop("bound", None)

        self.set_cost_function(cost_function)

        self.use_latex = use_latex if bool(find_executable("latex")) else False
        self._fitted = False

    def set_initial_values(
        self, initial_values: dict[str, Union[float, dict, Fitparam]] = {}
    ):  # sourcery skip: default-mutable-arg
        """Set the initial values of the fit parameters

        Args:
            initial_values (dict[str, Union[float, dict, Fitparam]], optional): The inital values the user sets. Defaults to {}.

        Raises:
            ValueError: If the initial values are not in the correct format
        """

        if not hasattr(self, "initial_values"):
            self.intial_values = self.func.guess(x=self._x.value, y=self._y.value)

        for key, value in initial_values.items():
            if isinstance(value, dict):
                if "value" in value:
                    self.intial_values[key].values = value["values"]
                if "limits" in value:
                    self.intial_values[key].limits = value["limits"]
                if "fixed" in value:
                    self.intial_values[key].fixed = value["fixed"]

            elif isinstance(value, (int, float)):
                self.intial_values[key] = Fitparam(values=value)

            else:
                raise ValueError("Initial value not supported")

        for key in list(self.intial_values.keys()):
            if key not in self.param_names:
                del self.intial_values[key]

        self._set_intial_values_only_from_inital_values()

    def _set_intial_values_only_from_inital_values(self):
        """Set the initial values of the fit parameters in the correct format for minuit"""
        self._initial_values_only: dict[str, Union[float, None]] = {
            key: value.values for key, value in self.intial_values.items() if key in self.param_names
        }

    def _set_minuit_with_values_limits_and_fixed(self):
        """Update the minuit object with the values, limits and fixed values from the initial values"""
        for p in self.param_names:
            if isinstance(self.intial_values[p], Fitparam):
                self.minuit.values[p] = self.intial_values[p].values
                self.minuit.limits[p] = self.intial_values[p].limits
                self.minuit.fixed[p] = self.intial_values[p].fixed

            else:
                self.minuit.values[p] = self.intial_values[p]

    def set_cost_function(self, cost_func):
        """Set the cost function to be used in the fit

        Args:
            cost_func (func): The cost function to be used in the fit

        Returns:
            self: The fitter object
        """
        if isinstance(cost_func, str):
            if cost_func == "chi2":
                self.cost_function = ExternalFunctions.Chi2Regression(
                    f=self.func.func,
                    x=self._x.value,
                    y=self._y.value,
                    sy=self._sy,
                    weights=self.weights,
                    bound=self.bound,
                )

            if cost_func == "chi2_no_uncertainty":
                self.cost_function = ExternalFunctions.Chi2Regression(
                    f=self.func.func,
                    x=self._x.value,
                    y=self._y.value,
                    weights=self.weights,
                    bound=self.bound,
                )

        else:
            self.cost_function = cost_func(
                f=self.func.func,
                x=self._x.value,
                y=self._y.value,
                sy=self._sy,
                weights=self.weights,
                bound=self.bound,
            )

        return self

    def set_minuit(self):
        """Set the minuit object with the correct initial values, limits and fixed values"""
        self.set_initial_values()
        self.minuit = Minuit(
            self.cost_function,
            **self._initial_values_only,
            name=[*self._initial_values_only.keys()],
        )
        self._set_minuit_with_values_limits_and_fixed()
