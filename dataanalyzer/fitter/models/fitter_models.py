# Author: Malthe Asmus Marciniak Nielsen
import copy
import inspect
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union
import numpy as np

from dataanalyzer.fitter.fitter_classsetup import *
from dataanalyzer.fitter.fitter_decorators import symbol_wrapper, unit_wrapper


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

        self._prefix: str = prefix
        self._params_guess: dict[str, Fitparam] = {}

        self.x_unit: str = kwargs.pop("x_unit", "")
        self.y_unit: str = kwargs.pop("y_unit", "")

        self.params_hint: dict = kwargs.pop("params_hint", {})
        self.independent_vars: list[str] = kwargs.pop("independent_vars", ["x"])
        self._param_root_names: list[str] = kwargs.pop("param_names", [])

        self._make_param_names()

    def __mul__(self, other):
        """Returns a new model that is the product of self and other.

        Args:
            other (ModelABC): The other model.

        Returns:
            ModelABC: The product of self and other.
        """
        if isinstance(other, int):
            return SumModel(*([self] * other))

    @abstractmethod
    def func(self, *args, **kwargs):
        """Function to be fitted. Must be implemented in subclass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abstractmethod
    def guess(
        self, x: Union[float, Iterable], y: Union[float, Iterable], *args, **kwargs
    ) -> dict[str, Fitparam]:
        """Guesses initial parameters for the fit. Must be implemented in subclass.

        Args:
            x (Union[float, Iterable]): The independent variable.
            y (Union[float, Iterable]): The dependent variable.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Fitparam]: A dict of the initial parameters."""
        pass

    @abstractmethod
    def funcname(self, *params):
        """Returns the name of the function to be fitted. Must be implemented in subclass.

        Args:
            *params (str): The parameters of the function.
        """
        pass

    @abstractmethod
    def units(self, x_unit, y_unit) -> dict[str, str]:
        """Returns the units of the parameters. Must be implemented in subclass.

        Args:
            x_unit (str): The unit of the independent variable.
            y_unit (str): The unit of the dependent variable.

        Returns:
            dict[str, str]: A dict of the units of the parameters.
        """
        if x_unit == "":
            x_unit = self.x_unit

        if y_unit == "":
            y_unit = self.y_unit

        return x_unit, y_unit

    @symbol_wrapper
    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
        """Returns the symbols of the parameters. Must be implemented in subclass.

        Returns:
            dict[str, str]: A dict of the symbols of the parameters.
        """
        return dict(zip(self._param_root_names, self._param_root_names))

    def _make_param_root_names(self):
        """Creates a list of the root names of the parameters."""
        # Create a dict of any positional parameters that have default values
        pos_params: dict[str, Optional[float]] = {}
        # Create a dict of the default values of the parameters
        self._def_values: dict[str, Optional[float]] = {}

        # If the function has an attribute "argnames" and "defaults"
        if hasattr(self.func, "argnames") and hasattr(self.func, "defaults"):
            # Add the argnames and defaults to the dict of positional parameters
            pos_params[self.func.argnames] = self.func.defaults
        else:
            # Otherwise, iterate over the positional arguments of the function
            for arg_name, arg_value in inspect.signature(self.func).parameters.items():
                # If the argument is positional or keyword (i.e. not *args or **kwargs),
                # and it has a default value
                if (
                    arg_value.kind == arg_value.POSITIONAL_OR_KEYWORD
                    and arg_value.default != arg_value.empty
                ):
                    # Add the argument name and default value to the dict of positional parameters
                    pos_params[arg_name] = arg_value.default

        # Iterate over the items of the dict of positional parameters
        for pos_key, pos_val in list(pos_params.items()):
            # If the key is None or the value is not an int or float
            if not (pos_key is None or isinstance(pos_val, (int, float))):
                # Remove the item from the dict of positional parameters
                pos_params.pop(pos_key)

            # If the key is in the list of independent variables
            if pos_key in self.independent_vars:
                # Remove the item from the dict of positional parameters
                pos_params.pop(pos_key)

        # Iterate over the keys in the dict of positional parameters
        for pos_param in list(pos_params.keys()):
            # If the key is not in the list of parameter root names
            if pos_param not in self._param_root_names:
                # Add the key to the list of parameter root names
                self._param_root_names.append(pos_param)

        # Add the dict of positional parameters to the dict of default values
        self._def_values = pos_params

    def _make_param_names(self):
        """Creates a list of the full names of the parameters."""
        self._make_param_root_names()
        self.param_names = [self._prefix + name for name in self._param_root_names]
        self._root2full = dict(zip(self._param_root_names, self.param_names))

    @property
    def param_symbols(self) -> dict[str, str]:
        """Returns the symbols of the parameters.

        Returns:
            dict[str, str]: A dict of the symbols of the parameters.
        """
        if not hasattr(self, "_param_symbols"):
            self.symbols()
        return self._param_symbols

    def _make_parameters(self, **kwargs) -> dict[str, Fitparam]:
        """Creates a dict of the parameters.

        Returns:
            dict[str, Fitparam]: A dict of the parameters.
        """
        if not self.param_names:
            self._make_param_names()

        params = {}

        for name in self.param_names:
            # Making empty parameter dict
            par = params.get(name, Fitparam())
            basename = name[len(self._prefix) :]

            # Updates parameter from guess
            if basename in self._def_values:
                par.values = self._def_values[basename]
            if par.values in (None, np.inf, -np.inf, np.nan):
                par.values = 0

            # Updates parameter from hint
            if basename in self.params_hint:
                par.values = self.params_hint[basename]

            # Updates parameter from kwargs (no prefix)
            if basename in kwargs:
                if isinstance(kwargs[basename], Fitparam):
                    par.update(kwargs[basename])
                elif isinstance(kwargs[basename], (int, float)):
                    par.values = kwargs[basename]

            # Updates parameter from kwargs (with prefix)
            if name in kwargs:
                if isinstance(kwargs[name], Fitparam):
                    par.update(kwargs[name])
                elif isinstance(kwargs[name], (int, float)):
                    par.values = kwargs[name]

            params[name] = par
        return params

    def _make_units(self, **kwargs) -> dict[str, str]:
        """Creates a dict of the units of the parameters.

        Args:
            **kwargs: The units of the parameters.

        Returns:
            dict[str, str]: A dict of the units of the parameters.
        """
        units = {}
        for name in self.param_names:
            basename = name[len(self._prefix) :]

            # Updates unit from kwargs (no prefix)
            if basename in kwargs:
                units[name] = kwargs[basename]

            # Updates unit from kwargs (with prefix)
            if name in kwargs:
                units[name] = kwargs[name]

        return units

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

    def __add__(self, other):
        """Adds two models.

        Args:
            other (ModelABC): The model to add.

        Returns:
            SumModel: The sum of the two models.
        """
        return SumModel(self, other)


####################################################################################################
#                   ABC Model Class __add__ function                                               #
####################################################################################################
class SumModel(ModelABC):
    """A class for the sum of models. The models are added in the order they are given.

    Args:
        *models (ModelABC): The models to add.
    """

    def __init__(self, *models):
        """Initializes the sum of models. The models are added in the order they are given.

        Args:
            *models (ModelABC): The models to add.
        """

        self.models = self._flatten_models(models)

        self.param_names: list[str] = []
        self._prefix: list[str] = []
        self._postfix: list[str] = ["" for _ in range(len(self.models))]

        self._root2full: dict[str, str] = {}
        self._params_guess: dict[str, Fitparam] = {}
        self._def_values: dict[str, Optional[float]] = {}
        self.params_hint: dict = {}
        self._param_root_names: list[str] = []

        for model in self.models:
            if isinstance(model._prefix, list):
                self._prefix += model._prefix
            else:
                self._prefix.append(model._prefix)

            self._root2full.update(model._root2full)
            self._params_guess.update(model._params_guess)

            self.params_hint.update(model.params_hint)

        self._set_param_root_name_and_get_prefix()
        self._make_parameters()
        self.symbols()

    def _flatten_models(self, models: list[ModelABC]) -> list[ModelABC]:
        """Flattens the models.

        Args:
            models (list[ModelABC]): The models to flatten.

        Returns:
            list[ModelABC]: The flattened models.
        """
        flattened_models = []
        for model in models:
            if isinstance(model, SumModel):
                print("Flattening SumModel")
                flattened_models += self._flatten_models(model.models)

            elif isinstance(model, ModelABC):
                flattened_models.append(model)

            else:
                raise TypeError(f"Model {model} is not a ModelABC or SumModel.")
        return flattened_models

    def _set_param_root_name_and_get_prefix(self):
        """Sets the root names of the parameters and gets the prefix of the parameters.

        Raises:
            ValueError: The models have the same parameter names.
        """
        # Get the root names of the parameters
        root_names_temp = [model._param_root_names for model in self.models]

        # Check if the models have the same parameter names
        for i, param_list in enumerate(root_names_temp):
            # If the models have the same parameter names
            if common_member(*root_names_temp):
                # Add the index of the model as a postfix to the parameters
                self._postfix[i] = f"_{i}"
                param_list = [f"{name}_{i}" for name in param_list]
            self._param_root_names += param_list

            # Set the parameter names
            limit = len(param_list)
            if hasattr(self.models[i], "poly_degree"):
                limit = self.models[i].poly_degree + 1

            self.models[i].param_names = param_list[:limit]
            self.param_names += [f"{self._prefix[i]}{name}" for name in param_list][
                :limit
            ]

    def func(self, x, *args, **kwargs):
        """Returns the sum of the models.

        Args:
            x (np.ndarray): Independent variable.

        Returns:
            np.ndarray: The sum of the models.
        """
        # Unpack the positional arguments into keyword arguments
        if args:
            kwargs = dict(zip(self.param_names, args))

        # Convert the independent variable to a numpy array
        x = np.array(x)

        # Initialize an array to store the sum of the model functions
        func_sum = np.zeros(x.shape)

        # Loop over the models
        for i, model in enumerate(self.models):
            # Get the postfix of the model
            p = self._postfix[i]

            # Create a dictionary of the keyword arguments for the model
            model_kwargs = {
                k.replace(p, ""): v for k, v in kwargs.items() if k in model.param_names
            }

            # Add the model function evaluated with the model-specific keyword arguments
            func_sum += model.func(x, **model_kwargs)

        return func_sum

    def guess(
        self, x: Union[float, Iterable], y: Union[float, Iterable], *args, **kwargs
    ):
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
        for i, model in enumerate(self.models):
            print(model.param_names)
            print("priny postfix", self._postfix)
            # Get the guessed parameters of the model
            model_guess = model.guess(x=x, y=y, *args, **kwargs)

            # Update the dictionary with the guessed parameters of the model
            guess |= model_guess

            # Get the postfix of the model
            p = self._postfix[i]

            # Update the guess of y by subtracting the model's guess from the
            # previous guess of y
            guess_values = {k.replace(p, ""): v.values for k, v in model_guess.items()}
            y_guess = model.func(x, **guess_values)
            y = y - y_guess

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
            func_name, func_str = (
                model.funcname(*args, **kwargs).replace("$", "").split(" = ")
            )

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
            units |= model.units(x=x, y=y)
        # Return the units dictionary
        return units

    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
        """Returns the symbols of the model.

        Returns:
            dict[str, str]: The symbols of the model.
        """
        # Initialize the symbols dictionary
        symbols = {}
        self._root2symbol = {}

        # Iterate over the models and update the symbols dictionary
        for model in self.models:
            symbols |= model.symbols(**symbols)
            self._root2symbol.update(model._root2symbol)
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
        for i, model in enumerate(self.models):
            # Get the postfix of the model
            p = self._postfix[i]

            # Create a dictionary of the keyword arguments for the model
            model_params = {
                k.replace(p, ""): v for k, v in params.items() if k in model.param_names
            }

            # Get the extrema of the model
            model_name = f"{model.__class__.__name__}{p}"
            extrema |= {model_name: model.get_extrema(model_params)}
        return extrema


####################################################################################################
#                   Linear Model                                                                   #
####################################################################################################
class LinearModel(ModelABC):
    """Linear Model Class for fitting.

    This class is a subclass of ModelABC. It is used to fit a linear function to data.

    Args:
        ModelABC (class): ModelABC class
    """

    def __init__(self, independent_vars=None, prefix="", **kwargs):
        """Initialize the LinearModel class.

        Args:
            independent_vars (list, optional): List of independent variables. Defaults to None.
            prefix (str, optional): Prefix for the model. Defaults to "".
        """
        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
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
        return self._make_parameters(slope=slope, intercept=intercept)

    def funcname(self, *params) -> str:
        """Function name.

        Returns:
            str: The function name.
        """
        if not params:
            params = self.param_symbols

        return rf"$\mathrm{{linear}}(x) = {params[0]} x + {params[1]}"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        """Units of the function.

        Args:
            x (Union[float, str, None]): If float, the value of the independent variable. If str, the unit of the independent variable. If None, the independent variable is dimensionless.
            y (Union[float, str, None]): If float, the value of the dependent variable. If str, the unit of the dependent variable. If None, the dependent variable is dimensionless.

        Returns:
            dict: The units or scaling factors of the parameters.
        """
        return {"slope": "y/x", "intercept": "y"}

    @symbol_wrapper
    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
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

    def __init__(self, independent_vars=None, prefix="", **kwargs):
        """Initialize the ProportionalModel class.

        Args:
            independent_vars (list, optional): List of independent variables. Defaults to None.
            prefix (str, optional): Prefix for the model. Defaults to "".
        """

        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
        super().__init__(**kwargs)

    def func(self, x, slope=1.0):
        x = np.array(x)
        return slope * x

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        slope = (y[-1] - y[0]) / (x[-1] - x[0])
        return self._make_parameters(slope=slope)

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_symbols

        return rf"$\mathrm{{proportional}}(x) = {params[0]} x"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"slope": "y/x"}

    @symbol_wrapper
    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
        return {"slope": "a"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        raise ValueError("Proportional model has no extrema")


####################################################################################################
#                   Gaussian Model                                                                 #
####################################################################################################
class GaussianModel(ModelABC):
    def __init__(self, independent_vars=None, prefix="", negative_peak=False, **kwargs):
        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
        self.negative_peak = negative_peak
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, center=0.0, sigma=1.0):
        x = np.array(x)
        return amplitude * np.exp(-(((x - center) / max(tiny, sigma)) ** 2))

    def guess(
        self, x: Union[float, Iterable], y: Union[float, Iterable], negative_peak=None
    ) -> dict:
        if negative_peak is None:
            negative_peak = self.negative_peak

        x, y = np.array(x), np.array(y)
        amplitude, center, sigma = guess_from_peak(y, x, negative_peak)
        return self._make_parameters(amplitude=amplitude, center=center, sigma=sigma)

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_symbols
        return rf"$\mathrm{{gaussian}}(x) = {params[0]} \exp(-((x - {params[1]}) / {params[2]})^2)"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "center": "x", "sigma": "x"}

    @symbol_wrapper
    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
        return {"amplitude": "A", "center": "μ", "sigma": "σ"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        return params["center"]


####################################################################################################
#                   Gaussian + Constant Model                                                      #
####################################################################################################
class GaussianConstantModel(ModelABC):
    def __init__(self, independent_vars=None, prefix="", negative_peak=False, **kwargs):
        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
        self.negative_peak = negative_peak
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, center=0.0, sigma=1.0, offset=0.0):
        x = np.array(x)
        return amplitude * np.exp(-(((x - center) / max(tiny, sigma)) ** 2)) + offset

    def guess(
        self, x: Union[float, Iterable], y: Union[float, Iterable], negative_peak=None
    ) -> dict:
        if negative_peak is None:
            negative_peak = self.negative_peak

        x, y = np.array(x), np.array(y)
        offset = np.mean(y)
        amplitude, center, sigma = guess_from_peak(y - offset, x, negative_peak)
        return self._make_parameters(
            amplitude=amplitude, center=center, sigma=sigma, offset=offset
        )

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_symbols
        return rf"$\mathrm{{gaussian}}(x) = {params[0]} \exp(-((x - {params[1]}) / {params[2]})^2) + {params[3]}"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "center": "x", "sigma": "x", "offset": "y"}

    @symbol_wrapper
    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
        return {"amplitude": "A", "center": "μ", "sigma": "σ", "offset": "c"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        return params["center"]


####################################################################################################
#                   Lorentzian Model                                                               #
####################################################################################################
class LorentzianModel(ModelABC):
    def __init__(self, independent_vars=None, prefix="", negative_peak=False, **kwargs):
        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
        self.negative_peak = negative_peak
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, center=0.0, sigma=1.0):
        x = np.array(x)
        return amplitude * sigma**2 / (max(tiny, sigma) ** 2 + (x - center) ** 2)

    def guess(
        self, x: Union[float, Iterable], y: Union[float, Iterable], negative_peak=None
    ) -> dict:
        if negative_peak is None:
            negative_peak = self.negative_peak

        x, y = np.array(x), np.array(y)
        amplitude, center, sigma = guess_from_peak(y, x, negative_peak, ampscale=1.25)
        return self._make_parameters(amplitude=amplitude, center=center, sigma=sigma)

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_symbols

        return rf"$\mathrm{{lorentzian}}(x) = \frac{{{params[0]}{params[2]}^2}}{{(x-{params[1]})^2+{params[2]}^2}}$"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "center": "x", "sigma": "x"}

    @symbol_wrapper
    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
        return {"amplitude": "A", "center": "μ", "sigma": "σ"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        return params["center"]


####################################################################################################
#                   Lorentzian + Constant Model                                                    #
####################################################################################################
class LorentzianConstantModel(ModelABC):
    def __init__(self, independent_vars=None, prefix="", negative_peak=False, **kwargs):
        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
        self.negative_peak = negative_peak
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, center=0.0, sigma=1.0, offset=0.0):
        x = np.array(x)
        return (
            amplitude * sigma**2 / (max(tiny, sigma) ** 2 + (x - center) ** 2)
            + offset
        )

    def guess(
        self, x: Union[float, Iterable], y: Union[float, Iterable], negative_peak=None
    ) -> dict:
        if negative_peak is None:
            negative_peak = self.negative_peak

        x, y = np.array(x), np.array(y)
        offset = np.mean(y)
        amplitude, center, sigma = guess_from_peak(y, x, negative_peak, ampscale=1.25)
        return self._make_parameters(
            amplitude=amplitude, center=center, sigma=sigma, offset=offset
        )

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_symbols

        return rf"$\mathrm{{lorentzian}}(x) = \frac{{{params[0]}{params[2]}^2}}{{(x-{params[1]})^2+{params[2]}^2}} + {params[3]}$"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "center": "x", "sigma": "x", "offset": "y"}

    @symbol_wrapper
    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
        return {"amplitude": "A", "center": "μ", "sigma": "σ", "offset": "c"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        return params["center"]


####################################################################################################
#                   Lorentzian + Polynomial Model                                                  #
####################################################################################################
class LorentzianPolynomialModel(ModelABC):
    def __init__(
        self,
        independent_vars=None,
        prefix="",
        degree: int = 3,
        negative_peak: bool = False,
        **kwargs,
    ):
        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}

        super().__init__(**kwargs)
        self.__model = PolynomialModel(degree=degree, **kwargs) + LorentzianModel(
            negative_peak=negative_peak, **kwargs
        )

        self.__dict__.update(self.__model.__dict__)

    def func(self, x, *args, **kwargs):
        return self.__model.func(x, *args, **kwargs)

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        return self.__model.guess(x, y)

    def funcname(self, *params) -> str:
        return self.__model.funcname(*params)

    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return self.__model.units(x, y)

    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
        return self.__model.symbols(**symbols)


####################################################################################################
#                   Split Lorentzian Model                                                         #
####################################################################################################
class SplitLorentzianModel(ModelABC):
    def __init__(self, independent_vars=None, prefix="", negative_peak=False, **kwargs):
        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
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

    def guess(
        self, x: Union[float, Iterable], y: Union[float, Iterable], negative_peak=None
    ) -> dict:
        if negative_peak is None:
            negative_peak = self.negative_peak

        x, y = np.array(x), np.array(y)
        amplitude, center, sigma = guess_from_peak(y, x, negative_peak, ampscale=1.25)
        return self._make_parameters(
            amplitude=amplitude, center=center, sigma=sigma, sigma_r=sigma
        )

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_symbols
        return (
            rf"$\mathrm{{split\ lorentzian}}(x) = \frac{{2 {params[0]} / ({params[2]} + {params[3]})}}"
            rf"{{({params[2]}^2 + (x - {params[1]})^2) + ({params[3]}^2 + (x - {params[1]})^2)}}$"
        )

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "center": "x", "sigma": "x", "split": "x"}

    @symbol_wrapper
    def symbols(self, *params) -> list[str]:
        return {"amplitude": "A", "center": "μ", "sigma": "σ", "split": "σ_r"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        return params["center"]


####################################################################################################
#                   Polynomial Model                                                               #
####################################################################################################
class PolynomialModel(ModelABC):
    def __init__(self, independent_vars=None, prefix="", degree: int = 3, **kwargs):
        self.poly_degree = degree
        kwargs["param_names"] = [f"c{i}" for i in range(degree + 1)]

        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
        super().__init__(**kwargs)

    def func(self, x, c0=0.0, c1=0.0, c2=0.0, c3=0.0, c4=0.0, c5=0.0, c6=0.0, c7=0.0):
        return np.polyval([c7, c6, c5, c4, c3, c2, c1, c0], x)

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        coeffs = np.polyfit(x, y, self.poly_degree)
        coeffs_dict = dict(zip(self.param_names, coeffs[::-1]))

        for i in range(self.poly_degree + 1, 8):
            coeffs_dict[f"c{i}"] = Fitparam(0.0, fixed=True)

        return self._make_parameters(**coeffs_dict)

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_symbols
        return rf"$\mathrm{{polynomial}}(x) = {params[0]} + {params[1]} x + {' + '.join(f'{params[i]} x^{i}' for i in range(2, self.poly_degree +1))}$"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        coeffs_units = {"c0": "y"}
        for i in range(1, self.poly_degree + 1):
            coeffs_units[f"c{i}"] = f"y / x**{i}"

        return coeffs_units

    @symbol_wrapper
    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
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

    def _find_mulitpolinomial_minima(self, params: dict):
        param_values = [p["value"] for p in params.values()][: self.poly_degree + 1][
            ::-1
        ]

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
    def __init__(self, independent_vars=None, prefix="", **kwargs):
        self.angular = kwargs.pop("angular", False)
        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, frequency=0.0, phi=0.0, offset=0.0):
        x = np.array(x)
        if not self.angular:
            frequency = frequency * 2 * np.pi
        return amplitude * np.sin(frequency * x + phi) + offset

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        [amplitude, frequency, phi, offset] = self._oscillations_guess(x, y)

        return self._make_parameters(
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
        freqs = fftpack.rfftfreq(len(x), d=(x[1] - x[0]) / (2 * np.pi))
        w = freqs[idx]
        f = w if self.angular else w / (2 * np.pi)

        dx = x[1] - x[0]
        indices_per_period = np.pi * 2 / w / dx
        std_window = round(indices_per_period)

        phi = np.angle(
            sum((y[:std_window] - c) * np.exp(-1j * (w * x[:std_window] - np.pi / 2)))
        )

        return [a, f, phi, c]

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_symbols
        f = f"{params[1]}" if self.angular else f"2π{params[1]}"
        return rf"$\mathrm{{oscillation}}(x) = {params[0]} \sin({f} x + {params[2]}) + {params[3]}$"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {
            "amplitude": "y",
            "frequency": "x**(-1)",
            "phi": "rad",
            "offset": "y",
        }

    @symbol_wrapper
    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
        return {
            "amplitude": "A",
            "frequency": "f",
            "phi": "φ",
            "offset": "c",
        }

    def get_period(self, params: dict) -> dict[str, float]:
        if self.angular:
            period = 1 / (2 * np.pi * params["frequency"]["value"])
            error = (
                params["frequency"]["error"]
                / params["frequency"]["value"] ** 2
                / (2 * np.pi)
            )
        else:
            period = 1 / params["frequency"]["value"]
            error = params["frequency"]["error"] / params["frequency"]["value"] ** 2

        return {"value": period, "error": error}


####################################################################################################
#                   Damped Oscillation Model                                                       #
####################################################################################################
class DampedOscillationModel(ModelABC):
    def __init__(self, independent_vars=None, prefix="", **kwargs):
        self.angular = kwargs.pop("angular", False)
        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, frequency=0.0, phi=0.0, decay=0.0, offset=0.0):
        x = np.array(x)
        if not self.angular:
            frequency = frequency * 2 * np.pi
        return amplitude * np.sin(frequency * x + phi) * np.exp(-x / decay) + offset

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        [amplitude, decay, frequency, phi, offset] = self._damped_oscillations_guess(
            x, y
        )

        return self._make_parameters(
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
        freqs = fftpack.rfftfreq(len(x), d=(x[1] - x[0]) / (2 * np.pi))
        w = freqs[idx]
        f = w if self.angular else w / (2 * np.pi)
        dx = x[1] - x[0]
        indices_per_period = np.pi * 2 / w / dx
        std_window = round(indices_per_period)
        initial_std = np.std(y[:std_window])
        noise_level = np.std(y[-2 * std_window :])
        for i in range(1, len(x) - std_window):
            std = np.std(y[i : i + std_window])
            if std < (initial_std - noise_level) * np.exp(-1):
                T = x[i]
                break
        p = np.arcsin((y[0] - c) / a)

        return [a, T, f, p, c]

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_symbols

        f = f"{params[1]}" if self.angular else f"2π{params[1]}"
        return rf"$\mathrm{{damped\ oscillation}}(x) = {params[0]} \sin({f} x + {params[2]}) \exp(-x / {params[3]}) + {params[4]}$"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {
            "amplitude": "y",
            "frequency": "x**(-1)",
            "phi": "rad",
            "decay": "x**(-1)",
            "offset": "y",
        }

    @symbol_wrapper
    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
        return {
            "amplitude": "A",
            "frequency": "f",
            "phi": "φ",
            "decay": "τ",
            "offset": "c",
        }


####################################################################################################
#                   Randomized Clifford Benchmark Model                                            #
####################################################################################################
class RandomizedCliffordBenchmarkModel(ModelABC):
    def __init__(self, independent_vars=None, prefix="", **kwargs):
        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
        super().__init__(**kwargs)

    def func(self, x, amplitude=1.0, phase=0.0, offset=0.0):
        x = np.array(x)
        return amplitude * phase**x + offset

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        amplitude = -y.min()
        phase = 0.99
        offset = y.min()
        return self._make_parameters(amplitude=amplitude, phase=phase, offset=offset)

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_symbols
        return rf"$\mathrm{{randomized\ clifford}}(x) = {params[0]} {params[1]} ^ x + {params[2]}$"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "phase": "", "offset": "y"}

    @symbol_wrapper
    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
        return {"amplitude": "A", "phase": "φ", "offset": "c"}


####################################################################################################
#                   Exponential Decay Model                                                        #
####################################################################################################
class ExponentialDecayModel(ModelABC):
    def __init__(self, independent_vars=None, prefix="", **kwargs):
        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
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

        offset = np.mean(
            y[round(len(y) * 0.9) :]
        )  # Use the last 10% of the data to determine the offset
        amplitude = (
            np.mean(y[:3]) - offset
        )  # Use the first 3 points to determine the amplitude
        decay = x[np.argmin(abs(y - (amplitude * np.exp(-1) + offset)))]

        return self._make_parameters(amplitude=amplitude, decay=decay, offset=offset)

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_symbols
        return rf"$\mathrm{{exponential\ decay}}(x) = {params[0]} \exp(-x / {params[1]}) + {params[2]}$"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "decay": "x", "offset": "y"}

    @symbol_wrapper
    def symbols(self, **symbols: dict[str, str]) -> dict[str, str]:
        return {"amplitude": "A", "decay": "τ", "offset": "c"}
