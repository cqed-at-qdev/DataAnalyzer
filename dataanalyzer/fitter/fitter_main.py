# Author: Malthe Asmus Marciniak Nielsen
from typing import Union

import numpy as np
from iminuit import Minuit, cost
from scipy import stats

from dataanalyzer.fitter import Fitparam
from dataanalyzer.utilities import Valueclass
from dataanalyzer.utilities.utilities import round_on_error, DataScaler


####################################################################################################
#                   Fitter Class                                                                   #
####################################################################################################
# class Fitter_old:
#     def __init__(
#         self,
#         func: ModelABC,
#         x: Union[Valueclass, list, tuple, np.ndarray],
#         y: Union[Valueclass, list, tuple, np.ndarray],
#         sy=None,
#         cost_function="chi2",
#         use_latex: bool = False,
#         **kwargs,
#     ) -> None:
#         """The fitter class

#         Args:
#             func (ModelABC): The model to be fitted
#             x (Union[Valueclass, list, tuple, np.ndarray]): The x values. Can be a Valueclass object or a list, tuple or numpy array
#             y (Union[Valueclass, list, tuple, np.ndarray]): The y values. Can be a Valueclass object or a list, tuple or numpy array
#             sy (np.ndarray, optional): The uncertainty of the y values. Defaults to None.
#             cost_function (str, optional): The cost function to be used in the fit. Defaults to "chi2".
#             use_latex (bool, optional): If the report should be in latex format. Defaults to True.

#         Raises:
#             ValueError: If the x and y values are not in the correct format
#         """
#         self.func = func
#         self.kwargs = kwargs

#         self._convert_and_scale_xy_data(x, y, sy)

#         self.set_initial_values(self.kwargs.pop("intial_values", {}))
#         self._set_initial_fitting_options(cost_function, use_latex)

#     ############# Main Functions ###################################################################
#     def _convert_data(self, x, y):
#         for data_name, data in zip(["x", "y"], [x, y]):
#             if not data:
#                 raise ValueError(f"{data_name} data is None")

#             setattr(self, data_name, data)
#             setattr(
#                 self,
#                 f"_{data_name}",
#                 Valueclass.fromfloat(copy.copy(data), f"{data_name} data"),
#             )

#     def _scale_data(self):
#         for data_name in ["x", "y"]:
#             scaled_data, unit_prefix, conversion_factor = convert_array_with_unit(getattr(self, f"_{data_name}").value)
#             getattr(self, f"_{data_name}").value = scaled_data
#             setattr(self, f"_{data_name}_unit_prefix", unit_prefix)
#             setattr(self, f"_{data_name}_cf", conversion_factor)

#     def _convert_and_scale_xy_data(self, x, y, sy=None):
#         self._convert_data(x, y)
#         self._ftt_xy_data()
#         self._scale_data()

#         self._sy = sy if sy is not None else self._y.copy().error
#         self._sy *= self._y_cf
#         self._conversion_factors = self.func.get_units(x=self._x_cf, y=self._y_cf)

#     def _ftt_xy_data(self):
#         if self._y.fft_type == "fft_y" and self._x.fft_type != "fft_x":
#             self._x = copy.deepcopy(self._x.fftfreq)
#         elif self._y.fft_type == "fft_x" and self._x.fft_type != "fft_y":
#             self._x = copy.deepcopy(self._x.fft)
#         elif self._x.fft_type == "fft_x" and self._y.fft_type != "fft_x":
#             self._y = copy.deepcopy(self._y.fftfreq)
#         elif self._x.fft_type == "fft_y" and self._y.fft_type != "fft_y":
#             self._y = copy.deepcopy(self._y.fft)

#     @property
#     def param_names(self):
#         """Returns the names of the fit parameters

#         Returns:
#             list: List of the names of the fit parameters
#         """
#         return self.func._full_name_list

#     @property
#     def initial_guess(self):
#         """Returns the initial guess of the fit parameters

#         Returns:
#             dict: Dictionary with the initial guess of the fit parameters
#         """
#         return {p: self.intial_values[p].values for p in self.param_names}

#     def do_fit(
#         self,
#         linspace_start=None,
#         linspace_stop=None,
#         linspace_steps=1000,
#         **kwargs,
#     ) -> tuple[np.ndarray, np.ndarray, dict, str]:
#         """Do the fit

#         Args:
#             return_linspace (bool, optional): If True a linspace of evalated points from linspace_start to linspace_stop of linspace_steps points is return. Defaults to True.
#             linspace_start (float, optional): The start of the linspace. Defaults to None. Linestart is set to the minimum of the x data if None
#             linspace_stop (float, optional): The stop of the linspace. Defaults to None. Linestop is set to the maximum of the x data if None
#             linspace_steps (int, optional): The number of points in the linspace. Defaults to 1000.

#         Returns:
#             Union[tuple[np.ndarray, np.ndarray, str], str]: If return_linspace is True a tuple with the linspace, the evaluated function and the latex string is returned. If return_linspace is False only the latex string is returned
#         """
#         self._do_fit()

#         return_list = []

#         if kwargs.pop("return_fit_array", True):
#             return_list += self.get_fit_array(linspace_start, linspace_stop, linspace_steps)

#         if kwargs.pop("return_params", True):
#             return_list.append(self.get_params())

#         if kwargs.pop("return_report", True):
#             return_list.append(self._report_string)

#         if kwargs.pop("return_guess_array", False):
#             return_list.append(self.get_guess_array(linspace_start, linspace_stop, linspace_steps))

#         return tuple(return_list)

#     def _do_fit(self):
#         self.set_minuit()
#         self.minuit.migrad()
#         self._calculate_prob_and_make_report()
#         self.params = self.get_params()
#         self._fitted = True

#     def _convert_x_and_y_to_valueclass(self, x_array, y_array):
#         if isinstance(self.x, Valueclass):
#             x_array = Valueclass(value=x_array, name=self.x.name, unit=self.x.unit)
#         if isinstance(self.y, Valueclass):
#             y_array = Valueclass(value=y_array, name=self.y.name, unit=self.y.unit)
#         return x_array / self._x_cf, y_array / self._y_cf

#     def _calculate_prob_and_make_report(self):
#         """Calculate the probability of the fit"""
#         self.nvar = len(self.minuit.parameters)  # Number of variables
#         self.ndof = len(self._x.value) - self.nvar  # type: ignore # Ndof = n data points - n variables
#         self.chi2 = self.minuit.fval  # The chi2 value
#         self.redchi2 = self.chi2 / self.ndof if self.chi2 else None  # The reduced chi2 value
#         self.prob = stats.chi2.sf(self.chi2, self.ndof)  # The chi2 probability
#         self._update_iminuit_report()

#     def _update_iminuit_report(self):
#         """Update the iminuit report"""
#         self._update_iminuit_report_with_function()
#         self._update_imuniut_report_with_parameters()
#         self._update_iminuit_report_with_statistics()

#     def _update_iminuit_report_with_function(self):
#         """Update the iminuit report with the function"""
#         funcname_str = f"Function: \n{self.func.funcname()}\n\n"
#         if len(funcname_str) > 100:
#             funcname_str = f"Function: \n{self.func.funcname()[:75]}...\n\n"
#         self._report_string = funcname_str

#     def _update_imuniut_report_with_parameters(self):
#         """Update the iminuit report with the parameters"""
#         poly_degree = getattr(self.func, "poly_degree", None)
#         values = self.minuit.values.to_dict().items()
#         errors = self.minuit.errors.to_dict().values()

#         unit_names = self._get_func_units()

#         self._report_string += "Fit parameters:\n"

#         for i, ((key, val), err) in enumerate(zip(values, errors)):
#             if key in self.param_names:
#                 unit = f"$\mathrm{{{unit_names[key]}}}$" if unit_names else ""
#                 name = f"${self.func._display_name_list[i]}$"
#                 self._add_parameter_string(unit, name, val, err)

#             if poly_degree is not None and i > poly_degree - 1:
#                 break

#     def _add_parameter_string(self, unit, key, val, err):
#         value_string = round_on_error(val, err, n_digits=2)
#         self._report_string += f"{key}: {value_string} {unit}\n"

#     def _update_iminuit_report_with_statistics(self):
#         """Update the iminuit report with the probability and the chi2 value"""
#         if any([self.chi2, self.ndof, self.prob]):
#             self._report_string += "\nFit statistics:\n"

#         if self.chi2 is not None:
#             self._report_string += f"Chi²: {np.format_float_scientific(self.chi2, precision=4)}\n"

#         if self.ndof is not None:
#             self._report_string += f"Degree of freedom: {self.ndof}\n"

#         if self.prob is not None:
#             self._report_string += f"Probability: {self.prob}\n"

#         self._report_string += "\n"

#     ############# Get Methods ######################################################################
#     def get_residuals(self, x=None, y=None):
#         x = self._x.value if x is None else x * self._x_cf
#         y = self._y.value if y is None else y * self._y_cf

#         residuals = (y - self.func.func(x=x, **self.minuit.values.to_dict())) / self._y_cf

#         return Valueclass(
#             value=residuals,
#             name=self._y.name,
#             unit=self._y.unit,
#             fft_type=self._y.fft_type,
#         )

#     def get_fit_array(self, linspace_start=None, linspace_stop=None, linspace_steps=1000):
#         x_fit = self._get_linspace_of_x(linspace_start, linspace_stop, linspace_steps)
#         y_fit = np.array(self.func.func(x=x_fit, **self.minuit.values.to_dict()))

#         x_fit, y_fit = self._convert_x_and_y_to_valueclass(x_fit, y_fit)
#         return x_fit, y_fit

#     def get_guess_array(self, linspace_start=None, linspace_stop=None, linspace_steps=1000):
#         x_guess = self._get_linspace_of_x(linspace_start, linspace_stop, linspace_steps)
#         y_guess = np.array(self.func.func(x=x_guess, **self.initial_guess))

#         x_guess, y_guess = self._convert_x_and_y_to_valueclass(x_guess, y_guess)
#         return x_guess, y_guess

#     def _get_linspace_of_x(self, linspace_start=None, linspace_stop=None, linspace_steps=1000):
#         if linspace_start is None:
#             linspace_start = np.min(self._x.value)
#         else:
#             linspace_start = linspace_start * self._x_cf

#         if linspace_stop is None:
#             linspace_stop = np.max(self._x.value)
#         else:
#             linspace_stop = linspace_stop * self._x_cf

#         return np.linspace(linspace_start, linspace_stop, linspace_steps)

#     def get_report(self):
#         """Get the report string"""
#         return self.minuit

#     def get_params(self, *param_names) -> dict:
#         """Get the parameters

#         Returns:
#             dict: A dictionary with the parameters
#         """
#         if not param_names:
#             param_names = self.param_names

#         return {p: self._get_converted_param(p) for p in param_names}

#     def _get_converted_param(self, param: str):
#         if param not in self._conversion_factors:
#             return {
#                 "value": self.minuit.values[param],
#                 "error": self.minuit.errors[param],
#             }

#         convertion_fator = self._conversion_factors[param]
#         value = self.minuit.values[param]
#         error = self.minuit.errors[param]

#         return {"value": value / convertion_fator, "error": error / convertion_fator}

#     def get_extrema(self):
#         if not hasattr(self, "params"):
#             self.do_fit()

#         return self.func.get_extrema(self.params)

#     def get_period(self):
#         if not hasattr(self, "params"):
#             self.do_fit()

#         return self.func.get_period(self.params)

#     def get_initial_values(self):
#         """Returns the initial values of the fit parameters

#         Returns:
#             dict: Dictionary with the initial values of the fit parameters
#         """
#         for p in self.param_names:
#             self.intial_values[p].values = self.minuit.values[p]
#             self.intial_values[p].limits = self.minuit.limits[p]
#             self.intial_values[p].fixed = self.minuit.fixed[p]
#         return self.intial_values

#     def _get_func_units(self) -> dict:
#         if not hasattr(self.func, "units"):
#             return {}

#         if self._x.unit is None or self._y.unit is None:
#             return {}

#         x_unit = self._x_unit_prefix + self._x.unit
#         y_unit = self._y_unit_prefix + self._y.unit

#         return self.func.get_units(x=x_unit, y=y_unit)  # type: ignore

#     ############# Set Methods ######################################################################
#     def set_symbols(self, **symbols: str):
#         self.func.set_symbols(**symbols)

#     def set_units(self, **units: Union[str, float, int]):
#         self.func.set_units(**units)

#     def set_parameters(self, **values: Union[Fitparam, dict, float, int]):
#         self.func.set_parameters(**values)

#     def _set_initial_fitting_options(self, cost_function, use_latex):
#         self.weights = self.kwargs.pop("weights", None)
#         self.bound = self.kwargs.pop("bound", None)

#         self.set_cost_function(cost_function)

#         self.use_latex = use_latex if bool(find_executable("latex")) else False
#         self._fitted = False

#     def set_initial_values(
#         self, initial_values: dict[str, Union[float, dict, Fitparam]] = {}
#     ):  # sourcery skip: default-mutable-arg
#         """Set the initial values of the fit parameters

#         Args:
#             initial_values (dict[str, Union[float, dict, Fitparam]], optional): The inital values the user sets. Defaults to {}.

#         Raises:
#             ValueError: If the initial values are not in the correct format
#         """

#         if not hasattr(self, "initial_values"):
#             self.intial_values = self.func.guess(x=self._x.value, y=self._y.value)

#         for key, value in initial_values.items():
#             if isinstance(value, dict):
#                 if "value" in value:
#                     self.intial_values[key].values = value["values"]
#                 if "limits" in value:
#                     self.intial_values[key].limits = value["limits"]
#                 if "fixed" in value:
#                     self.intial_values[key].fixed = value["fixed"]

#             elif isinstance(value, (int, float)):
#                 self.intial_values[key] = Fitparam(values=value)

#             else:
#                 raise ValueError("Initial value not supported")

#         for key in list(self.intial_values.keys()):
#             if key not in self.param_names:
#                 del self.intial_values[key]

#         self._set_intial_values_only_from_inital_values()

#     def _set_intial_values_only_from_inital_values(self):
#         """Set the initial values of the fit parameters in the correct format for minuit"""
#         self._initial_values_only: dict[str, Union[float, None]] = {
#             key: value.values for key, value in self.intial_values.items() if key in self.param_names
#         }

#     def _set_minuit_with_values_limits_and_fixed(self):
#         """Update the minuit object with the values, limits and fixed values from the initial values"""
#         for p in self.param_names:
#             if isinstance(self.intial_values[p], Fitparam):
#                 self.minuit.values[p] = self.intial_values[p].values
#                 self.minuit.limits[p] = self.intial_values[p].limits
#                 self.minuit.fixed[p] = self.intial_values[p].fixed

#             else:
#                 self.minuit.values[p] = self.intial_values[p]

#     def set_cost_function(self, cost_func):
#         """Set the cost function to be used in the fit

#         Args:
#             cost_func (func): The cost function to be used in the fit

#         Returns:
#             self: The fitter object
#         """
#         if isinstance(cost_func, str):
#             if cost_func == "chi2":
#                 self.cost_function = ExternalFunctions.Chi2Regression(
#                     f=self.func.func,
#                     x=self._x.value,
#                     y=self._y.value,
#                     sy=self._sy,
#                     weights=self.weights,
#                     bound=self.bound,
#                 )

#             if cost_func == "chi2_no_uncertainty":
#                 self.cost_function = ExternalFunctions.Chi2Regression(
#                     f=self.func.func,
#                     x=self._x.value,
#                     y=self._y.value,
#                     weights=self.weights,
#                     bound=self.bound,
#                 )

#             if cost_func == "chi2_iminuit":
#                 from iminuit.cost import LeastSquares

#                 self.cost_function = LeastSquares(
#                     x=self._x.value,
#                     y=self._y.value,
#                     yerror=self._sy,
#                     model=self.func.func,
#                 )

#             if cost_func == "UnbinnedNLL":
#                 from iminuit.cost import UnbinnedNLL, ExtendedUnbinnedNLL

#                 print("Using UnbinnedNLL as cost function")
#                 data = np.array([self._x.value, self._y.value])
#                 self.cost_function = ExtendedUnbinnedNLL(data=data, scaled_pdf=self.func.func)

#             if cost_func == "BinnedNLL":
#                 from iminuit.cost import BinnedNLL

#                 self.cost_function = BinnedNLL(
#                     n=self._y.value,
#                     xe=self._x.value,
#                     cdf=self.func.func,
#                 )
#         else:
#             self.cost_function = cost_func(
#                 self.func.func,
#                 self._x.value,
#                 self._y.value,
#                 sy=self._sy,
#                 weights=self.weights,
#                 bound=self.bound,
#             )

#         return self

#     def set_minuit(self):
#         """Set the minuit object with the correct initial values, limits and fixed values"""
#         self.set_initial_values()
#         self.minuit = Minuit(
#             self.cost_function,
#             **self._initial_values_only,
#             name=[*self._initial_values_only.keys()],
#         )
#         self._set_minuit_with_values_limits_and_fixed()


####################################################################################################
#                   Fitter Class (Refactoring)                                                     #
####################################################################################################
class Fitter:
    @property
    def inital_guess(self):
        if not hasattr(self, "_inital_guess"):
            self._inital_guess = self.model.guess(x=self.x.value, y=self.y.value)
        return self._inital_guess

    @inital_guess.setter
    def inital_guess(self, value):
        self._inital_guess = value

    @property
    def initial_values(self):
        return {key: value.values for key, value in self.inital_guess.items()}

    @property
    def initial_limits(self):
        return {key: value.limits for key, value in self.inital_guess.items()}

    @property
    def initial_fixed(self):
        return {key: value.fixed for key, value in self.inital_guess.items()}

    @property
    def scaled_initial_values(self):
        return {
            key: self.x_scaler.inverse_transform(value)
            for key, value in self.initial_values.items()
        }

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self.x_scaler = DataScaler()
        self._x = Valueclass.fromfloat(value, name="x data")
        self._x.value = self.x_scaler.fit(self._x.value)

    @property
    def x_scaled(self):
        return self.x_scaler.inverse_transform(self.x)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self.y_scaler = DataScaler()
        self._y = Valueclass.fromfloat(value, name="y data")
        self._y.value = self.y_scaler.fit(self._y.value)

    @property
    def y_scaled(self):
        return self.y_scaler.inverse_transform(self.y)

    @property
    def yerr(self):
        if (
            array_is_empty(self._yerr.value)
            and self._estimate_errors
            and self._estimate_errors_init
        ):
            self._yerr.value = 1e-10
            self._yerr.error = np.nan
        return self._yerr

    @yerr.setter
    def yerr(self, value):
        self._yerr = Valueclass.fromfloat(value, name="y error")
        self._estimate_errors = array_is_empty(self._yerr.value)
        self._yerr.value = self.y_scaler.transform(self._yerr.value)

    @property
    def yerr_scaled(self):
        return self.y_scaler.inverse_transform(self.yerr)

    @property
    def param_names(self):
        return self.model._full_name_list

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value):
        if isinstance(value, str):
            if value == "LeastSquares":
                self._cost = cost.LeastSquares(x=self.x.value, y=self.y.value, yerror=self.yerr.value, model=self.model.func)  # type: ignore
            elif value == "UnbinnedNLL":
                self._cost = cost.UnbinnedNLL(
                    data=(self.x.value, self.y.value), pdf=self.model.func
                )
            elif value == "BinnedNLL":
                self._cost = cost.BinnedNLL(n=self.y.value, xe=self.x.value, cdf=self.model.func)  # type: ignore
            else:
                raise ValueError(
                    "Cost function not supported. Supported cost functions are: 'LeastSquares', 'UnbinnedNLL', 'BinnedNLL'"
                )
        else:
            self._cost = value

    @property
    def param_conversion(self):
        return self.model.get_units(x=self.x_scaler.scale, y=self.y_scaler.scale)

    @property
    def parameters(self):
        """Return the parameters of the fit."""
        return self.get_parameters()

    @property
    def values(self):
        """Return the values of the parameters of the fit."""
        return self.get_values()

    @property
    def errors(self):
        """Return the errors of the parameters of the fit."""
        return self.get_errors()

    @property
    def parameter_units(self):
        """Return the units of the parameters of the fit."""
        if not self.x.has_unit or not self.y.has_unit:
            return {}  # No units available
        x_unit = self.x_scaler.unit_prefix + self.x.unit
        y_unit = self.y_scaler.unit_prefix + self.y.unit
        return self.model.get_units(x=x_unit, y=y_unit)

    @property
    def parameter_si_units(self):
        """Return the SI units of the parameters of the fit."""
        if not self.x.has_unit or not self.y.has_unit:
            return {}
        x_unit = self.x.unit
        y_unit = self.y.unit
        return self.model.get_units(x=x_unit, y=y_unit)

    def __init__(
        self,
        model,
        x,
        y,
        yerr=None,
        cost="LeastSquares",
        estimate_errors=True,
        **kwargs,
    ):
        self._estimate_errors_init = estimate_errors
        self._fitted = False

        self.model = model
        self.x = x
        self.y = y
        self.yerr = self.y.error if yerr is None else yerr
        self.cost = cost

        self.kwargs = kwargs
        self.kwargs["cost"] = cost

    def set_cost(self, cost):
        """Set the cost function to be used in the fit."""
        if cost is not None:
            self.cost = cost

    def set_symbols(self, **symbols: str):
        """Set the symbols of the parameters."""
        self.model.set_symbols(**symbols)

    def set_units(self, **units: Union[str, float, int]):
        """Set the units of the parameters."""
        self.model.set_units(**units)

    def set_parameters(self, **values: Union[Fitparam, dict, float, int]):
        """Set the values, limits and fixed values of the parameters."""
        self.model.set_parameters(**values)

    def set_initial_guess(
        self,
        **values: Union[
            Fitparam,
            dict,
            float,
            int,
        ],
    ):
        """Set the initial guess of the parameters."""
        for key, value in values.items():
            if isinstance(value, Fitparam):
                self.inital_guess[key] = value
            elif isinstance(value, dict):
                if "values" in value:
                    self.inital_guess[key].values = (
                        value["values"] * self.param_conversion[key]
                    )
                if "limits" in value:
                    self.inital_guess[key].limits = (
                        value["limits"] * self.param_conversion[key]
                    )
                if "fixed" in value:
                    self.inital_guess[key].fixed = value["fixed"]

            else:
                self.inital_guess[key].values = value * self.param_conversion[key]

    def _set_minuit_limits(self):
        """Set the limits of the parameters in the minuit object."""
        for key, value in self.initial_limits.items():
            if value is not None:
                self.minuit.limits[key] = value

    def _set_minuit_fixed(self):
        """Set the fixed values of the parameters in the minuit object."""
        for key, value in self.initial_fixed.items():
            if value is not None:
                self.minuit.fixed[key] = value

    def _set_minuit_limits_and_fixed(self):
        """Set the limits and fixed values of the parameters in the minuit object."""
        self._set_minuit_limits()
        self._set_minuit_fixed()

    def reset_cost(self):
        """Reset the cost function."""
        self.cost = self.kwargs.get("cost", "LeastSquares")

    def set_minuit(
        self, cost=None, inital_guess=None, parameter_names=None, reset=False, **kwargs
    ):
        """Set the minuit object with the correct initial values, limits and fixed values"""
        reset = self.kwargs.get("reset", reset)
        if (
            not reset
            and hasattr(self.model, "minuit")
            and self.model.minuit is not None
        ):
            return

        self.set_cost(cost)
        self.set_initial_guess(**inital_guess or {})
        self.set_symbols(**parameter_names or {})

        self.minuit = Minuit(
            self.cost, **self.initial_values, name=self.param_names, **kwargs
        )
        self._set_minuit_limits_and_fixed()

    def do_fit(self, **kwargs):
        """Fit the model to the data."""
        self._fitted = True
        self.set_minuit(**kwargs)
        self._estimate_errors_for_fit(**kwargs)  # Estimate the errors if needed
        self.minuit.migrad()

        return *self.get_fit_array(), self.get_parameters(), self.get_report()

    def _estimate_errors_for_fit(self, **kwargs):
        """Estimate the errors for the fit. (Only run once and if no errors are given.)"""
        if self._estimate_errors and self._estimate_errors_init:
            self._estimate_errors = False
            self.minuit.migrad()

            # Use the std of the residuals as an estimate of the error
            self.yerr.value = np.std(self.get_residuals(scaled=False)[1].value)

            self.reset_cost()
            kwargs.pop("reset", "")
            self.set_minuit(**kwargs, reset=True)

    def _check_if_fitted(self):
        """Check if the fit has been performed."""
        if not self._fitted:
            self.do_fit()

    def _get_array(self, params, x_min=None, x_max=None, n_points=1000):
        """Get the array for the given parameters."""
        x_min = self.x.min() if x_min is None else self.x_scaler.transform(x_min)
        x_max = self.x.max() if x_max is None else self.x_scaler.transform(x_max)
        x_array = np.linspace(x_min, x_max, n_points)

        xv, yv = self.x.copy(), self.y.copy()
        xv.value = self.x_scaler.inverse_transform(x_array)
        yv.value = self.y_scaler.inverse_transform(self.model.func(x=x_array, **params))
        return xv, yv

    def get_fit_array(self, x_min=None, x_max=None, n_points=1000):
        """Get the fit array."""
        self._check_if_fitted()
        return self._get_array(self.minuit.values.to_dict(), x_min, x_max, n_points)

    def get_guess_array(self, x_min=None, x_max=None, n_points=1000):
        """Get the guess array."""
        return self._get_array(self.initial_values, x_min, x_max, n_points)

    def get_residuals(self, x=None, y=None, scaled=True):
        """Get the residuals of the fit."""
        x = self.x_scaler.inverse_transform(self.x.value) if x is None else x
        y = self.y_scaler.inverse_transform(self.y.value) if y is None else y
        res = self.y_scaler.transform(y - self.model.func(x=x, **self.values))

        xv, resv = self.x.copy(), self.y.copy()
        xv.value = x if scaled else self.x_scaler.transform(x)
        resv.value = self.y_scaler.inverse_transform(res) if scaled else res
        return xv, resv

    def _get_parameter(self, param):
        """Get the value of a parameter."""
        if param not in self.param_names:
            raise ValueError(
                f"Parameter {param} not found. Available parameters are: {self.param_names}"
            )
        conversion = self.param_conversion[param]
        value = self.minuit.values[param]
        error = self.minuit.errors[param]
        return {"value": value / conversion, "error": error / conversion}

    def get_values(self, *params):
        """Get the values of the parameters."""
        params = params or self.param_names
        self._check_if_fitted()  # Fits the model if not already fitted
        return {param: self._get_parameter(param)["value"] for param in params}

    def get_errors(self, *params):
        """Get the values of the parameters."""
        params = params or self.param_names
        self._check_if_fitted()  # Fits the model if not already fitted
        return {param: self._get_parameter(param)["error"] for param in params}

    def get_parameters(self, *params):
        """Get the values of the parameters."""
        params = params or self.param_names
        self._check_if_fitted()  # Fits the model if not already fitted
        return {param: self._get_parameter(param) for param in params}

    def get_extrema(self):
        return self.model.get_extrema(self.parameters)

    def get_period(self):
        return self.model.get_period(self.parameters)

    def _calculate_probability(self):
        """Calculate the probability of the fit"""
        self.nvar = len(self.minuit.parameters)  # Number of variables
        self.ndof = len(self._x.value) - self.nvar  # type: ignore # Ndof = n data points - n variables
        self.chi2 = self.minuit.fval  # The chi2 value
        if self.ndof == 0 or not self.chi2:
            self.redchi2 = None
        else:
            self.redchi2 = self.chi2 / self.ndof  # The reduced chi2 value
        self.prob = stats.chi2.sf(self.chi2, self.ndof)  # The chi2 probability

    def get_probability(self):
        """Get the probability of the fit"""
        self._calculate_probability()
        return self.prob

    def get_report(self):
        """Get a report of the fit."""
        self.report = ""
        self._add_function_to_report()
        self._add_parameters_to_report()
        self._add_statistics_to_report()
        return self.report

    def _add_function_to_report(self):
        """Add the function to the report."""
        self.report += f"Function:\n{self.model.funcname()}\n\n"

    def _add_parameters_to_report(self):
        """Add the parameters to the report."""
        param_strs = [
            self._add_parameter_to_report(param) for param in self._get_print_params()
        ]
        self.report += f"Parameters:\n{''.join(param_strs)}\n"

    def _get_print_params(self):
        poly_degree = getattr(self.model, "poly_degree", None)
        print_params = []
        for i, param in enumerate(self.param_names):
            print_params.append(param)
            if poly_degree is not None and i > poly_degree - 1:
                break
        return print_params

    def _add_parameter_to_report(self, param):
        """Add a parameter to the report."""
        value = self.minuit.values[param]
        error = self.minuit.errors[param]
        return f"{param}: {round_on_error(value, error, n_digits=1)} {self._get_parameter_unit(param)}\n"

    def _get_parameter_unit(self, param):
        """Get the unit of a parameter."""
        if param not in self.parameter_units or not self.parameter_units[param]:
            return ""
        return f"${self.parameter_units[param]}$"

    def _add_statistics_to_report(self):
        """Add the statistics to the report."""
        if not hasattr(self, "chi2"):
            self._calculate_probability()

        self.report += "Fit statistics:\n"
        self.report += (
            f"Chi²: {np.format_float_scientific(self.chi2 or np.nan, precision=4)}\n"
        )
        self.report += f"Degree of freedom: {self.ndof}\n"
        self.report += f"Probability: {self.prob:.2f}\n\n"


def array_is_empty(a):
    return a.size == 0 or not a.all() or np.isnan(a).all()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataanalyzer.fitter.models import GaussianModel

    # Create model
    model = GaussianModel()

    # Make test data (gaussian)
    SCALE = 1e3
    AMPLITUDE = 10 * SCALE
    MEAN = 5 * SCALE
    SIGMA = 2 * SCALE
    N = 100
    NOISE = 0.7 * SCALE

    # Create data
    x = np.linspace(0, 10 * SCALE, N)
    y = model.func(
        x, amplitude=AMPLITUDE, center=MEAN, sigma=SIGMA
    ) + NOISE * np.random.normal(0, 1, N)

    x = Valueclass(x, name="testing x", unit="m")
    y = Valueclass(y, name="testing y", unit="Hz")  # , error=NOISE)

    # Fit model
    fit = Fitter(model, x, y, yerr=None)
    fit.do_fit()

    print("Yerr:", fit.yerr)
    print(fit.get_report())
