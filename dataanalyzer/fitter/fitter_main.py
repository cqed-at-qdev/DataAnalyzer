# Author: Malthe Asmus Marciniak Nielsen
from typing import Union

import numpy as np
from iminuit import Minuit, cost
from scipy import stats

from dataanalyzer.fitter import Fitparam
from dataanalyzer.utilities import Valueclass
from dataanalyzer.utilities.utilities import round_on_error, DataScaler


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
        return {key: self.x_scaler.inverse_transform(value) for key, value in self.initial_values.items()}

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self.x_scaler = DataScaler()
        self._x = Valueclass.fromfloat(value, name="x data")
        self._x.value = self.x_scaler.fit(self._x.value)
        self._x.error = self.x_scaler.transform(self._x.error)

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
        # self._y.error = self.y_scaler.transform(self._y.error)

    @property
    def y_scaled(self):
        return self.y_scaler.inverse_transform(self.y)

    @property
    def yerr(self):
        if array_is_empty(self._yerr.value) and self._estimate_errors and self._estimate_errors_init:
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
                self._cost = cost.UnbinnedNLL(data=(self.x.value, self.y.value), pdf=self.model.func)
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
                    self.inital_guess[key].values = value["values"] * self.param_conversion[key]
                if "limits" in value:
                    self.inital_guess[key].limits = value["limits"] * self.param_conversion[key]
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

    def set_minuit(self, cost=None, inital_guess=None, parameter_names=None, reset=False, **kwargs):
        """Set the minuit object with the correct initial values, limits and fixed values"""
        reset = self.kwargs.get("reset", reset)
        if not reset and hasattr(self.model, "minuit") and self.model.minuit is not None:
            return

        self.set_cost(cost)
        self.set_initial_guess(**inital_guess or {})
        self.set_symbols(**parameter_names or {})

        self.minuit = Minuit(self.cost, **self.initial_values, name=self.param_names, **kwargs)
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

    def get_array(self, x=None):
        self._check_if_fitted()
        x = self.x_scaler.inverse_transform(self.x.value) if x is None else x
        # res = self.y_scaler.transform(self.model.func(x=x, **self.values))
        # return self.y_scaler.inverse_transform(res)
        return x, self.model.func(x=x, **self.values)

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
            raise ValueError(f"Parameter {param} not found. Available parameters are: {self.param_names}")
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

    def get_extrema_y(self):
        return self.model.get_extrema_y(self.parameters)

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
        param_strs = [self._add_parameter_to_report(param) for param in self._get_print_params()]
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
        self.report += f"Chi²: {np.format_float_scientific(self.chi2 or np.nan, precision=4)}\n"
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
    y = model.func(x, amplitude=AMPLITUDE, center=MEAN, sigma=SIGMA) + NOISE * np.random.normal(0, 1, N)

    x = Valueclass(x, name="testing x", unit="m")
    y = Valueclass(y, name="testing y", unit="Hz")  # , error=NOISE)

    # Fit model
    fit = Fitter(model, x, y, yerr=None)
    fit.do_fit()

    print("Yerr:", fit.yerr)
    print(fit.get_report())
