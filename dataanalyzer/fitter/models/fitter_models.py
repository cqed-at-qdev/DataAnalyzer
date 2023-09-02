# Author: Malthe Asmus Marciniak Nielsen
import copy
import inspect
from abc import ABC, abstractmethod
from typing import Iterable, Union
import numpy as np
import scipy

from dataanalyzer.fitter.fitter_classsetup import *
from dataanalyzer.utilities.utilities import convert_unit_to_str_or_float


####################################################################################################
#                   ABC Model Class                                                                #
####################################################################################################
class ModelABC(ABC):
    def __init__(self, prefix: str = "", **kwargs):
        """Abstract base class for all models. Must be subclassed.

        Args:
            prefix (str, optional): The prefix of the model. Defaults to "".
            **kwargs: Additional keyword arguments.
        """

        self._name = (self.__class__.__name__,)
        self._prefix = prefix
        self._suffix = kwargs.pop("suffix", "")

    def _make_inital_parameters(self):
        self._parameters = [
            Fitparam(base_name=arg_name, values=arg_value.default, model=self)
            for arg_name, arg_value in inspect.signature(self.func).parameters.items()
            if (arg_value.kind == arg_value.POSITIONAL_OR_KEYWORD and arg_value.default != arg_value.empty)
        ]

    ############# Abstract Methods #################################################################
    @abstractmethod
    def func(self, *args, **kwargs):
        """Function to be fitted. Must be implemented in subclass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abstractmethod
    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable], *args, **kwargs) -> dict[str, Fitparam]:
        """Guesses initial parameters for the fit. Must be implemented in subclass.

        Args:
            x (Union[float, Iterable]): The independent variable.
            y (Union[float, Iterable]): The dependent variable.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Fitparam]: A list of the parameters."""
        pass

    @abstractmethod
    def funcname(self, *params) -> str:
        """Returns the name of the function to be fitted. Must be implemented in subclass.

        Args:
            *params (str): The parameters of the function.
        """
        pass

    @property
    @abstractmethod
    def units(self) -> dict[str, str]:
        """Returns the units of the parameters. Must be implemented in subclass.

        Args:
            x_unit (str): The unit of the independent variable.
            y_unit (str): The unit of the dependent variable.

        Returns:
            dict[str, str]: A dict of the units of the parameters.
        """
        pass

    @property
    @abstractmethod
    def symbols(self) -> dict[str, str]:
        """Returns the symbols of the parameters. Must be implemented in subclass.

        Returns:
            dict[str, str]: A dict of the symbols of the parameters.
        """
        pass

    ############# Properties #######################################################################
    @property
    def parameters(self):
        if not hasattr(self, "_parameters"):
            self._make_inital_parameters()
        return self._parameters

    @property
    def _base_name_list(self):
        return [param.base_name for param in self.parameters]

    @property
    def _full_name_list(self):
        return [param.full_name for param in self.parameters]

    @property
    def _display_name_list(self):
        return [param.display_name for param in self.parameters]

    ############# Get and Set Methods ##############################################################
    def _set_genereal_method(self, value_type: str, **new_params):
        """Updates the parameters of the model.

        Args:
            method (str): The method to use for updating the parameters.
            **params: The parameters to update.
        """
        for new_param in new_params:
            if new_param in self._full_name_list:
                setattr(
                    self.parameters[self._full_name_list.index(new_param)],
                    value_type,
                    new_params[new_param],
                )

            elif new_param in self._base_name_list:
                setattr(
                    self.parameters[self._base_name_list.index(new_param)],
                    value_type,
                    new_params[new_param],
                )

            else:
                raise ValueError(f"Parameter {new_param} not found. Available parameters are {self._full_name_list}")

    def set_symbols(self, **symbols: str):
        """Sets the symbols of the parameters.

        Args:
            **symbols: The symbols to set.
        """
        self._set_genereal_method("symbol", **symbols)

    def set_units(self, **units: Union[str, float, int]):
        """Sets the units of the parameters.

        Args:
            **units: The units to set.
        """
        self._set_genereal_method("unit", **units)

    def set_parameters(self, **values: Union[Fitparam, dict, float, int]):
        """Sets the values of the parameters.

        Args:
            **values: The values to set.
        """
        for v, k in values.items():
            if isinstance(k, Fitparam):
                for value_type in ["values", "errors", "limits", "fixed"]:
                    self._set_genereal_method(value_type, **{v: getattr(k, value_type)})
            elif isinstance(k, dict):
                for value_type in ["values", "errors", "limits", "fixed"]:
                    self._set_genereal_method(value_type, **{v: k[value_type]})
            elif isinstance(k, (float, int)):
                self._set_genereal_method("values", **{v: k})
            else:
                raise TypeError(
                    f"Type [{type(k)}] not supported. Use Fitparam, dict or float. Supported types are [{Fitparam, dict, float}]"
                )

        for param in self.parameters:
            if param.values in (None, np.inf, -np.inf, np.nan):
                param.values = 0

    def _get_parameters_as_dict(self, **params) -> dict[str, Fitparam]:
        """Returns a dict of the parameters.

        Args:
            **params: The parameters to update.

        Returns:
            list[Fitparam]: A list of the parameters.
        """
        self.set_parameters(**params)
        return {param.full_name: param for param in self.parameters}

    def get_units(self, x, y) -> dict[str, Union[float, str]]:
        """Returns a dict of the units of the parameters.

        Args:
            x (Union[float, Iterable]): The independent variable.
            y (Union[float, Iterable]): The dependent variable.

        Returns:
            dict[str, Union[float, str]]: A dict of the units of the parameters.
        """
        return {
            parameter.full_name: convert_unit_to_str_or_float(f=parameter.unit, x=x, y=y)
            for parameter in self.parameters
        }  # type: ignore

    def get_extrema(self, params: dict) -> dict:
        """Returns the extrema of the model.

        Args:
            params (dict): Fitted parameters of the model.

        Raises:
            NotImplementedError: get_extrema not implemented for this model.

        Returns:
            dict: The extrema of the model.
        """
        raise NotImplementedError("get_extrema not implemented for this model")

    def get_extrema_y(self, params: dict) -> dict:
        """Returns the function value at the extrema of the model.

        Args:
            params (dict): Fitted parameters of the model.

        Raises:
            NotImplementedError: get_extrema not implemented for this model.

        Returns:
            dict: The extrema of the model.
        """
        # TODO: implement default using values from get_extrema()
        raise NotImplementedError("get_extrema_y not implemented for this model")

    def get_period(self, params: dict) -> float:
        """Returns the period of the model.

        Args:
            params (dict): Fitted parameters of the model.

        Raises:
            NotImplementedError: get_period not implemented for this model.

        Returns:
            float: The period of the model.
        """
        raise NotImplementedError("get_period not implemented for this model")

    ############# Methods ##########################################################################
    def __add__(self, other) -> "SumModel":
        """Adds two models.

        Args:
            other (ModelABC): The model to add.

        Returns:
            SumModel: The sum of the two models.
        """
        return SumModel(self, other)

    def __mul__(self, other) -> "SumModel":
        """Returns a new model that is the product of self and other.

        Args:
            other (ModelABC): The other model.

        Returns:
            ModelABC: The product of self and other.
        """
        if not isinstance(other, int):
            raise TypeError("Can only multiply model with int")

        self_copy = copy.copy(self)
        for _ in range(other - 1):
            self_copy += copy.copy(self)
        return SumModel(self_copy)  # type: ignore


####################################################################################################
#                   ABC Model Class __add__ function                                               #
####################################################################################################
class SumModel(ModelABC):
    """A class for the sum of models. The models are added in the order they are given.

    Args:
        *models (ModelABC): The models to add.
    """

    def __init__(self, *models: ModelABC):
        """Initializes the sum of models. The models are added in the order they are given.

        Args:
            *models (ModelABC): The models to add.
        """

        self.models = self._flatten_models(models)
        self._add_sufixes()

    def _flatten_models(self, models: Union[tuple[ModelABC], list[ModelABC]]) -> list[ModelABC]:
        """Flattens the models.

        Args:
            models (list[ModelABC]): The models to flatten.

        Returns:
            list[ModelABC]: The flattened models.
        """
        flattened_models = []
        for model in models:
            if isinstance(model, SumModel):
                flattened_models += self._flatten_models(model.models)

            elif isinstance(model, ModelABC):
                flattened_models += [model]

            else:
                raise TypeError(f"Model {model} is not a ModelABC or SumModel.")
        return flattened_models

    def _add_sufixes(self):
        if duplicats_list := [x for x in self._base_name_list if self._base_name_list.count(x) > 1]:
            for i, model in enumerate(self.models, start=1):
                if all(item in duplicats_list for item in model._base_name_list):
                    model._suffix = f"_{i}"

    def func(self, x, *args, **kwargs):
        """Returns the sum of the models.

        Args:
            x (np.ndarray): Independent variable.

        Returns:
            np.ndarray: The sum of the models.
        """
        # Unpack the positional arguments into keyword arguments
        if args:
            kwargs = dict(zip(self._full_name_list, args))

        # Convert the independent variable to a numpy array
        x = np.array(x)

        # Initialize an array to store the sum of the model functions
        func_sum = np.zeros(x.shape)

        # Loop over the models
        for model in self.models:
            # Create a dictionary of the keyword arguments for the model
            model_kwargs = {}
            for k, v in kwargs.items():
                if k in model._full_name_list:
                    k_index = model._full_name_list.index(k)
                    model_kwargs[model._base_name_list[k_index]] = v

            # Add the model function evaluated with the model-specific keyword arguments
            func_sum += model.func(x, **model_kwargs)

        return func_sum

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable], *args, **kwargs):
        """Returns the guessed parameters of the model.

        Args:
            x (Union[float, Iterable]): Independent variable.
            y (Union[float, Iterable]): Dependent variable.

        Returns:
            dict: The guessed parameters of the model.
        """
        # Convert x, y to numpy arrays
        x, y = np.array(x), np.array(y)

        # Initialize a dictionary to store the guessed parameters
        guess = {}
        # Iterate through all models
        for model in self.models:
            # Get the guessed parameters of the model
            model_guess = model.guess(x=x, y=y, *args, **kwargs)

            # Update the dictionary with the guessed parameters of the model
            guess |= model_guess

            # Update the guess of y by subtracting the model's guess from the
            # previous guess of y
            guess_values = {v.base_name: v.values for k, v in model_guess.items()}
            y_guess = model.func(x, **guess_values)
            y -= y_guess

        return guess

    def funcname(self, *args, **kwargs):
        """Returns the function name of the model.

        Returns:
            str: The function name of the model.
        """
        # Initialize the function names and function strings
        func_names, func_strs = "", ""

        # Loop over the models
        for model in self.models:
            # Get the function name and the function string
            func_name, func_str = model.funcname(*args, **kwargs).replace("$", "").split(" = ")

            # Add the function name to the function names
            func_names += f"{func_name} +"
            # Add the function string to the function strings
            func_strs += f"{func_str} +"

        # Return the function names and function strings
        return f"${func_names[:-1]} = {func_strs[:-1]}$"

    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        """Returns the units of the model.

        Args:
            x (Union[float, str, None]): Independent variable.
            y (Union[float, str, None]): Dependent variable.

        Returns:
            dict: The units of the model.
        """
        # Initialize the units dictionary
        units = {}

        # Iterate over the models and update the units dictionary
        for model in self.models:
            units |= model.get_units(x=x, y=y)
        # Return the units dictionary
        return units

    @property
    def symbols(self) -> dict[str, str]:
        """Returns the symbols of the model.

        Returns:
            dict[str, str]: The symbols of the model.
        """
        # Initialize the symbols dictionary
        symbols = {}

        # Iterate over the models and update the symbols dictionary
        for model in self.models:
            symbols.update(**model.symbols)
        return symbols

    def get_extrema(self, params: dict) -> dict[str, dict[str, float]]:
        """Returns the extrema of the model.

        Args:
            params (dict): The parameters of the model.

        Returns:
            dict[str, dict[str, float]]: The extrema of the model.
        """
        # Initialize the extrema dictionary
        extrema = {}

        # Iterate over the models and update the extrema dictionary
        for model in self.models:
            # Get the postfix of the model
            p = model._suffix

            # Create a dictionary of the keyword arguments for the model
            model_params = {k.replace(p, ""): v for k, v in params.items() if k in model._full_name_list}

            # Get the extrema of the model
            model_name = f"{model.__class__.__name__}{p}"
            extrema |= {model_name: model.get_extrema(model_params)}
        return extrema

    @property
    def parameters(self):
        if not hasattr(self, "_parameters"):
            self._parameters = []
            for model in self.models:
                self._parameters += model.parameters

        return self._parameters


####################################################################################################
#                   Linear Model                                                                   #
####################################################################################################
class LinearModel(ModelABC):
    """Linear Model Class for fitting.

    This class is a subclass of ModelABC. It is used to fit a linear function to data.

    Args:
        ModelABC (class): ModelABC class
    """

    def __init__(self, **kwargs):
        """Initialize the LinearModel class.

        Args:
            independent_vars (list, optional): List of independent variables. Defaults to None.
            prefix (str, optional): Prefix for the model. Defaults to "".
        """
        super().__init__(**kwargs)

    def func(self, x, slope=1.0, intercept=0.0):
        """Function to fit.

        Args:
            x (array): Independent variable.
            slope (float, optional): The slope of the line. Defaults to 1.0.
            intercept (float, optional): The intercept of the line. Defaults to 0.0.

        Returns:
            array: The fitted function.
        """
        x = np.array(x)
        return slope * x + intercept

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        """Guess the parameters of the function.

        Args:
            x (Union[float, Iterable]): The independent variable.
            y (Union[float, Iterable]): The dependent variable.

        Returns:
            dict: The guessed parameters.
        """
        x, y = np.array(x), np.array(y)
        slope, intercept = np.polyfit(x, y, 1)
        return self._get_parameters_as_dict(slope=slope, intercept=intercept)

    def funcname(self, *params) -> str:
        """Function name.

        Returns:
            str: The function name.
        """
        if not params:
            params = self._display_name_list

        return rf"$f(x) = {params[0]} x + {params[1]}$"

    @property
    def units(self) -> dict[str, str]:
        """Units of the function.

        Args:
            x (Union[float, str, None]): If float, the value of the independent variable. If str, the unit of the independent variable. If None, the independent variable is dimensionless.
            y (Union[float, str, None]): If float, the value of the dependent variable. If str, the unit of the dependent variable. If None, the dependent variable is dimensionless.

        Returns:
            dict: The units or scaling factors of the parameters.
        """
        return {"slope": "y/x", "intercept": "y"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"slope": "a", "intercept": "b"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        raise ValueError("Linear model has no extrema")


####################################################################################################
#                   Proportional Model                                                             #
####################################################################################################
class ProportionalModel(ModelABC):
    """Proportional Model Class for fitting.

    This class is a subclass of ModelABC. It is used to fit a proportional function to data.

    Args:
        ModelABC (class): ModelABC class
    """

    def __init__(self, **kwargs):
        """Initialize the ProportionalModel class.

        Args:
            independent_vars (list, optional): List of independent variables. Defaults to None.
            prefix (str, optional): Prefix for the model. Defaults to "".
        """

        super().__init__(**kwargs)

    def func(self, x, slope=1.0):
        x = np.array(x)
        return slope * x

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        slope = (y[-1] - y[0]) / (x[-1] - x[0])
        return self._get_parameters_as_dict(slope=slope)

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list

        return rf"$f(x) = {params[0]} x$"

    @property
    def units(self) -> dict[str, str]:
        return {"slope": "y/x"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"slope": "a"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        raise ValueError("Proportional model has no extrema")


####################################################################################################
#                   Gaussian Model                                                                 #
####################################################################################################
class GaussianModel(ModelABC):
    def __init__(self, negative_peak=False, **kwargs):
        super().__init__(**kwargs)

        self.negative_peak = negative_peak

    def func(self, x, amplitude=1.0, center=0.0, sigma=1.0):
        x = np.array(x)
        return amplitude * np.exp(-(((x - center) / max(tiny, sigma)) ** 2))

    def guess(
        self,
        x: Union[float, Iterable],
        y: Union[float, Iterable],
        negative_peak=None,
    ) -> dict[str, Fitparam]:
        if negative_peak is None:
            negative_peak = self.negative_peak

        x, y = np.array(x), np.array(y)
        amplitude, center, sigma = guess_from_peak(y, x, negative_peak)

        return self._get_parameters_as_dict(amplitude=amplitude, center=center, sigma=sigma)

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list
        return rf"$f(x) = {params[0]} \exp(-((x - {params[1]}) / {params[2]})^2)$"

    @property
    def units(self) -> dict[str, str]:
        return {"amplitude": "y", "center": "x", "sigma": "x"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"amplitude": "A", "center": "μ", "sigma": "σ"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        return params["center"]


####################################################################################################
#                   Gaussian + Constant Model                                                      #
####################################################################################################
class GaussianConstantModel(ModelABC):
    def __init__(self, negative_peak=False, **kwargs):
        self.negative_peak = negative_peak
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, center=0.0, sigma=1.0, offset=0.0):
        x = np.array(x)
        return amplitude * np.exp(-(((x - center) / max(tiny, sigma)) ** 2)) + offset

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable], negative_peak=None) -> dict:
        if negative_peak is None:
            negative_peak = self.negative_peak

        x, y = np.array(x), np.array(y)
        offset = np.mean(y)
        amplitude, center, sigma = guess_from_peak(y - offset, x, negative_peak)
        return self._get_parameters_as_dict(amplitude=amplitude, center=center, sigma=sigma, offset=offset)

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list
        return rf"$f(x) = {params[0]} \exp(-((x - {params[1]}) / {params[2]})^2) + {params[3]}$"

    @property
    def units(self) -> dict[str, str]:
        return {"amplitude": "y", "center": "x", "sigma": "x", "offset": "y"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"amplitude": "A", "center": "μ", "sigma": "σ", "offset": "c"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        return params["center"]


####################################################################################################
#                   Lorentzian Model                                                               #
####################################################################################################
class LorentzianModel(ModelABC):
    def __init__(self, negative_peak=False, **kwargs):
        self.negative_peak = negative_peak
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, center=0.0, sigma=1.0):
        x = np.array(x)
        return amplitude * sigma**2 / (max(tiny, sigma) ** 2 + (x - center) ** 2)

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable], negative_peak=None) -> dict:
        if negative_peak is None:
            negative_peak = self.negative_peak

        x, y = np.array(x), np.array(y)
        amplitude, center, sigma = guess_from_peak(y, x, negative_peak, ampscale=1.25)
        return self._get_parameters_as_dict(amplitude=amplitude, center=center, sigma=sigma)

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list

        return rf"$f(x) = \frac{{{params[0]}{params[2]}^2}}{{(x-{params[1]})^2+{params[2]}^2}}$"

    @property
    def units(self) -> dict[str, str]:
        return {"amplitude": "y", "center": "x", "sigma": "x"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"amplitude": "A", "center": "μ", "sigma": "σ"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        return params["center"]


####################################################################################################
#                   Lorentzian + Constant Model                                                    #
####################################################################################################
class LorentzianConstantModel(ModelABC):
    def __init__(self, negative_peak=False, **kwargs):
        self.negative_peak = negative_peak
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, center=0.0, sigma=1.0, offset=0.0):
        x = np.array(x)
        return amplitude * sigma**2 / (max(tiny, sigma) ** 2 + (x - center) ** 2) + offset

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable], negative_peak=None) -> dict:
        if negative_peak is None:
            negative_peak = self.negative_peak

        x, y = np.array(x), np.array(y)
        offset = np.mean(y)
        amplitude, center, sigma = guess_from_peak(y, x, negative_peak, ampscale=1.25)
        return self._get_parameters_as_dict(amplitude=amplitude, center=center, sigma=sigma, offset=offset)

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list

        return rf"$f(x) = \frac{{{params[0]}{params[2]}^2}}{{(x-{params[1]})^2+{params[2]}^2}} + {params[3]}$"

    @property
    def units(self) -> dict[str, str]:
        return {"amplitude": "y", "center": "x", "sigma": "x", "offset": "y"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"amplitude": "A", "center": "μ", "sigma": "σ", "offset": "c"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        return params["center"]


####################################################################################################
#                   Lorentzian + Polynomial Model                                                  #
####################################################################################################
class LorentzianPolynomialModel(ModelABC):
    def __init__(
        self,
        degree: int = 3,
        negative_peak: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = PolynomialModel(degree=degree, **kwargs) + LorentzianModel(negative_peak=negative_peak, **kwargs)

        self.__dict__.update(self._model.__dict__)

    def func(self, x, *args, **kwargs):
        return self._model.func(x, *args, **kwargs)

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        return self._model.guess(x, y)

    def funcname(self, *params) -> str:
        return self._model.funcname(*params)

    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return self._model.units(x, y)

    @property
    def symbols(self) -> dict[str, str]:
        return self._model.symbols


####################################################################################################
#                   Split Lorentzian Model                                                         #
####################################################################################################
class SplitLorentzianModel(ModelABC):
    def __init__(self, negative_peak=False, **kwargs):
        self.negative_peak = negative_peak
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, center=0.0, sigma=1.0, sigma_r=0.0):
        x = np.array(x)
        s = max(tiny, sigma)
        r = max(tiny, sigma_r)
        ss = s * s
        rr = r * r
        xc2 = (x - center) ** 2
        amp = 2 * amplitude / (np.pi * (s + r))
        return amp * (ss * (x < center) / (ss + xc2) + rr * (x >= center) / (rr + xc2))

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable], negative_peak=None) -> dict:
        if negative_peak is None:
            negative_peak = self.negative_peak

        x, y = np.array(x), np.array(y)
        amplitude, center, sigma = guess_from_peak(y, x, negative_peak, ampscale=1.25)
        return self._get_parameters_as_dict(amplitude=amplitude, center=center, sigma=sigma, sigma_r=sigma)

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list
        return (
            rf"$f(x) = \frac{{2 {params[0]} / ({params[2]} + {params[3]})}}$"
            rf"${{({params[2]}^2 + (x - {params[1]})^2) + ({params[3]}^2 + (x - {params[1]})^2)}}$"
        )

    @property
    def units(self) -> dict[str, str]:
        return {"amplitude": "y", "center": "x", "sigma": "x", "split": "x"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"amplitude": "A", "center": "μ", "sigma": "σ", "split": "σ_r"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        return params["center"]


####################################################################################################
#                   Polynomial Model                                                               #
####################################################################################################
class PolynomialModel(ModelABC):
    def __init__(self, degree: int = 3, **kwargs):
        self.poly_degree = degree
        super().__init__(**kwargs)

    def func(self, x, c0=0.0, c1=0.0, c2=0.0, c3=0.0, c4=0.0, c5=0.0, c6=0.0, c7=0.0):
        return np.polyval([c7, c6, c5, c4, c3, c2, c1, c0], x)

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        coeffs = np.polyfit(x, y, self.poly_degree)
        coeffs_dict = dict(zip(self._full_name_list, coeffs[::-1]))

        for i in range(self.poly_degree + 1, 8):
            coeffs_dict[f"c{i}"] = Fitparam(values=0.0, fixed=True)

        return self._get_parameters_as_dict(**coeffs_dict)

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list
        return rf"$f(x) = {params[0]} + {params[1]} x + {' + '.join(f'{params[i]} x^{i}' for i in range(2, self.poly_degree +1))}$"

    @property
    def units(self) -> dict[str, str]:
        coeffs_units = {"c0": "y"}
        for i in range(1, self.poly_degree + 1):
            coeffs_units[f"c{i}"] = f"y / x**{i}"

        return coeffs_units

    @property
    def symbols(self) -> dict[str, str]:
        return {f"c{i}": f"c_{i}" for i in range(self.poly_degree + 1)}

    def get_extrema(self, params: dict) -> dict[str, float]:
        if self.poly_degree == 1:
            raise ValueError("Cannot find extrema for a linear function")
        elif self.poly_degree == 2:
            from uncertainties import ufloat

            c1 = ufloat(params["c1"]["value"], params["c1"]["error"])
            c2 = ufloat(params["c2"]["value"], params["c2"]["error"])

            extrema = -c1 / (2 * c2)  # type: ignore
            return {"value": extrema.n, "error": extrema.s}
        else:
            return self._find_mulitpolinomial_minima(params)

    def get_extrema_y(self, params: dict) -> dict[str, float]:
        if self.poly_degree == 1:
            raise ValueError("Cannot find extrema for a linear function")
        elif self.poly_degree == 2:
            from uncertainties import ufloat

            c0 = ufloat(params["c0"]["value"], params["c0"]["error"])
            c1 = ufloat(params["c1"]["value"], params["c1"]["error"])
            c2 = ufloat(params["c2"]["value"], params["c2"]["error"])

            extrema = c0 - c1**2 / (4 * c2)
            return {"value": extrema.n, "error": extrema.s}
        else:
            return self._find_mulitpolinomial_minima(params)

    def _find_mulitpolinomial_minima(self, params: dict):
        param_values = [p["value"] for p in params.values()][: self.poly_degree + 1][::-1]

        np_poly = np.poly1d(param_values)  # Note: polynomials are opposite order

        roots = np.real(np.roots(np_poly.deriv()))
        maxima = roots[np.argmax(np_poly(roots))]

        print(
            "Warning: extrema for polynomial model is not implemented yet",
            "Returning the maximum of the first derivative",
            "Unsertainties are not calculated, but sqrt(maxima) is returned",
        )

        return {"value": maxima, "error": np.sqrt(maxima)}


####################################################################################################
#                   Oscillation Model                                                              #
####################################################################################################
class OscillationModel(ModelABC):
    def __init__(self, **kwargs):
        self.angular = kwargs.pop("angular", False)
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, frequency=0.0, phi=0.0, offset=0.0):
        x = np.array(x)
        if not self.angular:
            frequency = frequency * 2 * np.pi
        return amplitude * np.sin(frequency * x + phi) + offset

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        [amplitude, frequency, phi, offset] = self._oscillations_guess(x, y)

        return self._get_parameters_as_dict(
            amplitude=amplitude,
            frequency=frequency,
            phi=phi,
            offset=offset,
        )

    def _oscillations_guess(self, x, y):
        # Adapted from QDev wrappers, `qdev_fitter`
        from scipy import fftpack

        a = (y.max() - y.min()) / 2
        c = y.mean()
        yhat = fftpack.rfft(y - y.mean())
        idx = (yhat**2).argmax()
        dx = np.diff(x).mean()
        freqs = fftpack.rfftfreq(len(x), d=dx / (2 * np.pi))
        w = freqs[idx]
        f = w if self.angular else w / (2 * np.pi)

        indices_per_period = np.pi * 2 / w / dx

        std_window = round(indices_per_period)

        phi = np.angle(sum((y[:std_window] - c) * np.exp(-1j * (w * x[:std_window] - np.pi / 2))))

        return [a, f, phi, c]

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list
        f = f"{params[1]}" if self.angular else f"2π{params[1]}"
        return rf"$f(x) = {params[0]} \sin({f} x + {params[2]}) + {params[3]}$"

    @property
    def units(self) -> dict[str, str]:
        return {
            "amplitude": "y",
            "frequency": "x**(-1)",
            "phi": "rad",
            "offset": "y",
        }

    @property
    def symbols(self) -> dict[str, str]:
        return {
            "amplitude": "A",
            "frequency": "f",
            "phi": "φ",
            "offset": "c",
        }

    def get_period(self, params: dict) -> dict[str, float]:
        if self.angular:
            period = 1 / (2 * np.pi * params["frequency"]["value"])
            error = params["frequency"]["error"] / params["frequency"]["value"] ** 2 / (2 * np.pi)
        else:
            period = 1 / params["frequency"]["value"]
            error = params["frequency"]["error"] / params["frequency"]["value"] ** 2

        return {"value": period, "error": error}


####################################################################################################
#                   Damped Oscillation Model                                                       #
####################################################################################################
class DampedOscillationModel(ModelABC):
    def __init__(self, **kwargs):
        self.angular = kwargs.pop("angular", False)
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, frequency=0.0, phi=0.0, decay=0.0, offset=0.0):
        x = np.array(x)
        if not self.angular:
            frequency = frequency * 2 * np.pi
        return amplitude * np.sin(frequency * x + phi) * np.exp(-x / decay) + offset

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        [amplitude, decay, frequency, phi, offset] = self._damped_oscillations_guess(x, y)

        return self._get_parameters_as_dict(
            amplitude=amplitude,
            frequency=frequency,
            phi=phi,
            decay=decay,
            offset=offset,
        )

    def _damped_oscillations_guess(self, x, y):
        # Adapted from QDev wrappers, `qdev_fitter`
        from scipy import fftpack

        a = (y.max() - y.min()) / 2
        c = y.mean()
        T = x[round(len(x) / 2)]
        yhat = fftpack.rfft(y - y.mean())
        idx = (yhat**2).argmax()
        dx = np.diff(x).mean()
        freqs = fftpack.rfftfreq(len(x), d=dx / (2 * np.pi))
        w = freqs[idx]
        f = w if self.angular else w / (2 * np.pi)

        indices_per_period = np.pi * 2 / w / dx
        std_window = round(indices_per_period)
        initial_std = np.std(y[:std_window])
        noise_level = np.std(y[-2 * std_window :])
        for i in range(1, len(x) - std_window):
            std = np.std(y[i : i + std_window])
            if std < (initial_std - noise_level) * np.exp(-1):
                T = x[i]
                break

        # print(f"y[0]: {y[0]}, c: {c}, a: {a}, f: {f}, T: {T}")
        p = np.arcsin((y[0] - c) / a)

        if np.isnan(p):
            p = 0

        return [a, T, f, p, c]

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list

        f = f"{params[1]}" if self.angular else f"2π{params[1]}"
        return rf"$f(x) = {params[0]} \sin({f} x + {params[2]}) \exp(-x / {params[3]}) + {params[4]}$"

    @property
    def units(self) -> dict[str, str]:
        return {
            "amplitude": "y",
            "frequency": "x**(-1)",
            "phi": "rad",
            "decay": "x",
            "offset": "y",
        }

    @property
    def symbols(self) -> dict[str, str]:
        return {
            "amplitude": "A",
            "frequency": "f",
            "phi": "φ",
            "decay": "τ",
            "offset": "c",
        }


####################################################################################################
#                   Sum of Oscillations Model                                                       #
####################################################################################################

# TODO: Finish sum of oscillation model for ddd repetitions of rabi pulses mvf Chris


class SumOscillationModelOdd(ModelABC):
    def __init__(self, **kwargs):
        self.angular = kwargs.pop("angular", False)
        self.N = kwargs.pop("N", 1)
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, frequency=0.0, phi=0.0, offset=0.0):
        x = np.array(x)
        N = self.N
        if not self.angular:
            frequency = frequency * 2 * np.pi
        temp = np.zeros(x.shape)
        for i in np.arange(1, 2 * N + 1, 2):
            temp += a * np.cos(i * freq * x - phi) + offset
        return temp / (N + 1)

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        [amplitude, frequency, phi, offset] = self._oscillations_guess(x, y)

        return self._get_parameters_as_dict(amplitude=amplitude, frequency=frequency, phi=phi, offset=offset)

    def _oscillations_guess(self, x, y):
        # Adapted from QDev wrappers, `qdev_fitter`
        from scipy import fftpack

        a = (y.max() - y.min()) / 2
        c = y.mean()
        # yhat = fftpack.rfft(y - y.mean())
        idx = y.argmax()
        # freqs = fftpack.rfftfreq(len(x), d=(x[1] - x[0]) / (2 * np.pi))
        w = 2 / x[idx]
        f = w if self.angular else w / (2 * np.pi)

        dx = x[1] - x[0]
        indices_per_period = np.pi * 2 / w / dx
        std_window = round(indices_per_period)

        phi = f / 2

        return [a, f, phi, c]

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list
        f = f"{params[1]}" if self.angular else f"2π{params[1]}"
        return rf"$f(x) = {params[0]} \sin({f} x + {params[2]}) + {params[3]}$"

    @property
    def units(self) -> dict[str, str]:
        return {
            "amplitude": "y",
            "frequency": "x**(-1)",
            "phi": "rad",
            "offset": "y",
        }

    @property
    def symbols(self) -> dict[str, str]:
        return {
            "amplitude": "A",
            "frequency": "f",
            "phi": "φ",
            "offset": "c",
        }

    def get_period(self, params: dict) -> dict[str, float]:
        if self.angular:
            period = 1 / (2 * np.pi * params["frequency"]["value"])
            error = params["frequency"]["error"] / params["frequency"]["value"] ** 2 / (2 * np.pi)
        else:
            period = 1 / params["frequency"]["value"]
            error = params["frequency"]["error"] / params["frequency"]["value"] ** 2

        return {"value": period, "error": error}


####################################################################################################
#                   Randomized Clifford Benchmark Model                                            #
####################################################################################################
class RandomizedCliffordBenchmarkModel(ModelABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, phase=0.0, offset=0.0):
        x = np.array(x)
        return amplitude * phase**x + offset

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        amplitude = -y.min()
        phase = 0.99
        offset = y.min()
        return self._get_parameters_as_dict(amplitude=amplitude, phase=phase, offset=offset)

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list
        return rf"$f(x) = {params[0]} {params[1]} ^ x + {params[2]}$"

    @property
    def units(self) -> dict[str, str]:
        return {"amplitude": "y", "phase": "", "offset": "y"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"amplitude": "A", "phase": "φ", "offset": "c"}


####################################################################################################
#                   Exponential Decay Model                                                        #
####################################################################################################
class ExponentialDecayModel(ModelABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, decay=1.0, offset=0.0):
        x = np.array(x)
        decay = not_zero(decay)
        return amplitude * np.exp(-x / decay) + offset

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        # offset = np.min(y[len(y)]) - 1e-4  # Small error added  TODO: why is this small error added?
        # decay = (x[-1] - x[0]) / -np.log((y[-1] - offset) / (y[0] - offset))
        # amplitude = (y[0] - offset) / np.exp(-x[0] / decay)

        offset = np.mean(y[round(len(y) * 0.9) :])  # Use the last 10% of the data to determine the offset
        amplitude = np.mean(y[:3]) - offset  # Use the first 3 points to determine the amplitude
        decay = x[np.argmin(abs(y - (amplitude * np.exp(-1) + offset)))]

        return self._get_parameters_as_dict(amplitude=amplitude, decay=decay, offset=offset)

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list
        return rf"$f(x) = {params[0]} \exp(-x / {params[1]}) + {params[2]}$"

    @property
    def units(self) -> dict[str, str]:
        return {"amplitude": "y", "decay": "x", "offset": "y"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"amplitude": "A", "decay": "τ", "offset": "c"}


####################################################################################################
#                   RB Decay Model                                                                 #
####################################################################################################
class RBDecayModel(ModelABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RB Decay"

    def func(self, x, amplitude=1.0, p=1.0, offset=0.0):
        x = np.array(x)
        p = not_zero(p)
        return amplitude * p**x + offset

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)

        offset = np.mean(y[round(len(y) * 0.9) :])  # Use the last 10% of the data to determine the offset
        amplitude = np.mean(y[:3]) - offset  # Use the first 3 points to determine the amplitude
        p = x[np.argmin(abs(y - (amplitude * np.exp(-1) + offset)))]
        p = 1 - 1 / p

        return self._get_parameters_as_dict(amplitude=amplitude, p=p, offset=offset)

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list
        return rf"$f(x) = {params[0]} {params[1]}^x + {params[2]}$"

    @property
    def units(self) -> dict[str, str]:
        return {"amplitude": "y", "p": "x", "offset": "y"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"amplitude": "A", "p": "p", "offset": "c"}


####################################################################################################
#                   Gaussian Multiple Model                                                        #
####################################################################################################
class GaussianMultipleModel(ModelABC):
    def __init__(self, n_peaks: Union[str, int] = "auto", negative_peak=False, **kwargs):
        super().__init__(**kwargs)
        self.n_peaks = n_peaks
        self.negative_peak = negative_peak
        self._kwargs = kwargs

        if isinstance(n_peaks, int):
            self._model = GaussianModel(negative_peak=negative_peak, **kwargs) * n_peaks
            self.__dict__.update(self._model.__dict__)
        else:
            print(
                "Warning: n_peaks is not set, using default value 'auto'.",
                "The function 'guess' must be called before fitting, to set the number of peaks.",
            )

    def func(self, x, *args, **kwargs):
        return self._model.func(x, *args, **kwargs)

    def guess(
        self,
        x: Union[float, Iterable],
        y: Union[float, Iterable],
        n_peaks: Union[str, int, None] = None,
        negative_peak=None,
    ) -> dict:
        if n_peaks is None:
            n_peaks = self.n_peaks
        if negative_peak is None:
            negative_peak = self.negative_peak

        x, y = np.array(x), np.array(y)

        guess = guess_from_multipeaks(y=y, x=x, n_peaks=n_peaks, negative=negative_peak)

        if n_peaks == "auto":
            self.n_peaks = len(guess)
            self._model = GaussianModel(negative_peak=negative_peak, **self._kwargs) * self.n_peaks
            self.__dict__.update(self._model.__dict__)

        return self._get_parameters_as_dict(**guess)

    def funcname(self, *params) -> str:
        return self._model.funcname(*params)

    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return self._model.units(x, y)

    @property
    def symbols(self) -> dict[str, str]:
        return self._model.symbols


####################################################################################################
#                   Lorentzian Multiple Model                                                      #
####################################################################################################
class LorentzianMultipleModel(ModelABC):
    def __init__(self, n_peaks: Union[str, int] = "auto", negative_peak=False, **kwargs):
        super().__init__(**kwargs)
        self.n_peaks = n_peaks
        self.negative_peak = negative_peak
        self._kwargs = kwargs

        if isinstance(n_peaks, int):
            self._model = LorentzianModel(negative_peak=negative_peak, **kwargs) * n_peaks
            self.__dict__.update(self._model.__dict__)
        else:
            print(
                "Warning: n_peaks is not set, using default value 'auto'.",
                "The function 'guess' must be called before fitting, to set the number of peaks.",
            )

    def func(self, x, *args, **kwargs):
        return self._model.func(x, *args, **kwargs)

    def guess(
        self,
        x: Union[float, Iterable],
        y: Union[float, Iterable],
        n_peaks: Union[str, int, None] = None,
        negative_peak=None,
    ) -> dict:
        if n_peaks is None:
            n_peaks = self.n_peaks
        if negative_peak is None:
            negative_peak = self.negative_peak

        x, y = np.array(x), np.array(y)

        guess = guess_from_multipeaks(y=y, x=x, n_peaks=n_peaks, negative=negative_peak)

        if n_peaks == "auto":
            self.n_peaks = len(guess)
            self._model = LorentzianModel(negative_peak=negative_peak, **self._kwargs) * self.n_peaks
            self.__dict__.update(self._model.__dict__)

        return self._get_parameters_as_dict(**guess)

    def funcname(self, *params) -> str:
        return self._model.funcname(*params)

    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return self._model.units(x, y)

    @property
    def symbols(self) -> dict[str, str]:
        return self._model.symbols


####################################################################################################
#                   Constant Model                                                                 #
####################################################################################################
class ConstantModel(ModelABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def func(self, x, offset=0.0):
        return np.ones(len(x)) * offset

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        offset = np.mean(y)
        return self._get_parameters_as_dict(offset=offset)

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list
        return rf"$f(x) = {params[0]}$"

    @property
    def units(self) -> dict[str, str]:
        return {"offset": "y"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"offset": "c"}


####################################################################################################
#                   Sinc Model                                                                     #
####################################################################################################
class SincModel(ModelABC):
    def __init__(self, negative_peak=False, **kwargs):
        self.negative_peak = negative_peak
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, xb=1.0, x0=1.0, offset=1.0):
        x = np.array(x)
        return amplitude * np.sinc(np.pi * (x - xb) / x0) + offset

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        offset = np.mean(y)

        if self.negative_peak:
            amplitude = np.min(y) - offset
            center = x[np.argmin(y)]
        else:
            amplitude = np.max(y) - offset
            center = x[np.argmax(y)]

        # Get peak width from the first derivative
        _, _, sigma = guess_from_peak(y, x, self.negative_peak)

        return self._get_parameters_as_dict(
            amplitude=amplitude,
            xb=center,
            x0=sigma.values * 4,
            offset=offset,
        )

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list

        return (
            rf"$f(x) = {params[0]} \cdot \mathrm{{sinc}}(\frac{{\pi(x - {params[1]})}}{{{params[2]}}}) + {params[3]}$"
        )

    @property
    def units(self) -> dict[str, str]:
        return {"amplitude": "y", "offset": "y", "xb": "x", "x0": "x"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"amplitude": "A", "offset": "c", "xb": "x_b", "x0": "x_0"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        return params["xb"]


####################################################################################################
#                   Frequency Spectrum Model                                                       #
####################################################################################################
class FreqSpectrumModel(ModelABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def func(self, x, v_per_phi0=1, flux_offset=0, max_freq=1, Ec=1):
        """Function to fit the frequency spectrum of a qubit.

        Args:
            x (array): Input data (in V).
            v_per_phi0 (float): Period of frequency spectrum (in V)
            flux_offset (float): Flux offset (in units of Phi_0)
            max_freq (float): Maximum frequency
            Ec (float): Charging energy

        Returns:
            array: f(x)
        """

        d = 0  # turn this into a fit parameter for asymmetric transmons (0<d<1 for asymmetric transmons)

        x = np.array(x)
        return (max_freq + Ec) * (
            d**2 + (1 - d**2) * (np.cos(np.pi * (x / v_per_phi0 + flux_offset))) ** 2
        ) ** 0.25 - Ec

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict[str, Fitparam]:
        x, y = np.array(x), np.array(y)

        max_freq = np.max(y)
        v_per_phi0 = 500
        flux_offset = -150 / v_per_phi0
        Ec = -0.274

        return self._get_parameters_as_dict(v_per_phi0=v_per_phi0, flux_offset=flux_offset, max_freq=max_freq, Ec=Ec)

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list
        return rf"$f(x) = ({params[2]} + {params[3]}) \cdot (d^2 + (1 - d^2) \cdot (\cos(\pi \cdot (\frac{{x}}{{{params[0]}}} + {params[1]})))^2)^{{1/4}} - {params[3]}$"

    @property
    def units(self) -> dict[str, str]:
        return {"v_per_phi0": "x", "max_freq": "y", "flux_offset": "", "Ec": "y"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"v_per_phi0": "V_{\Phi_0}", "max_freq": "f_{max}", "flux_offset": "\Phi_{offset}", "Ec": "E_C"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        return -params["flux_offset"] * params["v_per_phi0"]

    def get_extrema_y(self, params: dict) -> dict[str, float]:
        return params["max_freq"]


####################################################################################################
#                   cos^4 + cos^2sin^2 + sin^4 model                                               #
####################################################################################################
class C4CS2S4Model(ModelABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def func(self, x, c_0=1.0, c_1=1.0, c_2=1.0):
        x = np.array(x)
        return c_0 * np.cos(x / 2) ** 4 + c_1 * np.cos(x / 2) ** 2 * np.sin(x / 2) ** 2 + c_2 * np.sin(x / 2) ** 4

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        c_0 = y[0]
        c_2 = y[-1]
        c_1 = 4 * y[len(y) // 2] - c_0 - c_2

        return self._get_parameters_as_dict(
            c_0=c_0,
            c_1=c_1,
            c_2=c_2,
        )

    def funcname(self, *params) -> str:
        if not params:
            params = self._display_name_list

        return rf"$f(x) = {params[0]} \mathrm{{cos}}^4(x/2) + {params[1]} \mathrm{{cos}}^2(x/2)\mathrm{{sin}}^2(x/2) + {params[2]} \mathrm{{sin}}^4(x/2) $"

    @property
    def units(self) -> dict[str, str]:
        return {"c_0": "y", "c_1": "y", "c_2": "y"}

    @property
    def symbols(self) -> dict[str, str]:
        return {"c_0": "c_0", "c_1": "c_1", "c_2": "c_2"}
