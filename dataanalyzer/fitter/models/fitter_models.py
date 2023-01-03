# Author: Malthe Asmus Marciniak Nielsen
import copy
import inspect
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union

import numpy as np

from dataanalyzer.fitter.fitter_classsetup import *
from dataanalyzer.fitter.fitter_decorators import unit_wrapper


####################################################################################################
#                   ABC Model Class                                                                #
####################################################################################################
class ModelABC(ABC):
    def __init__(self, prefix: str = "", **kwargs):
        self.param_names: list[str] = []
        self._prefix: str = prefix
        self._root2full: dict[str, str] = {}
        self._params_guess: dict[str, Fitparam] = {}

        self.x_unit: str = kwargs.pop("x_unit", "")
        self.y_unit: str = kwargs.pop("y_unit", "")

        self.params_hint: dict = kwargs.pop("params_hint", {})
        self.independent_vars: list[str] = kwargs.pop("independent_vars", ["x"])
        self._param_root_names: list[str] = kwargs.pop("param_names", [])

        self._make_param_names()

    @abstractmethod
    def func(self, *args, **kwargs):
        pass

    @abstractmethod
    def guess(
        self, x: Union[float, Iterable], y: Union[float, Iterable], *args, **kwargs
    ) -> dict[str, Fitparam]:
        pass

    @abstractmethod
    def funcname(self, *args, **kwargs):
        pass

    @abstractmethod
    def funcname_latex(self, *args, **kwargs):
        pass

    @abstractmethod
    def units(self, x_unit, y_unit) -> dict[str, str]:
        if x_unit == "":
            x_unit = self.x_unit

        if y_unit == "":
            y_unit = self.y_unit

        return x_unit, y_unit

    def _make_param_root_names(self):
        pos_params: dict[str, Optional[float]] = {}
        self._def_values: dict[str, Optional[float]] = {}

        if hasattr(self.func, "argnames") and hasattr(self.func, "defaults"):
            pos_params[self.func.argnames] = self.func.defaults
        else:
            for arg_name, arg_value in inspect.signature(self.func).parameters.items():
                if (
                    arg_value.kind == arg_value.POSITIONAL_OR_KEYWORD
                    and arg_value.default != arg_value.empty
                ):
                    pos_params[arg_name] = arg_value.default

        for pos_key, pos_val in list(pos_params.items()):
            if not (pos_key is None or isinstance(pos_val, (int, float))):
                pos_params.pop(pos_key)

            if pos_key in self.independent_vars:
                pos_params.pop(pos_key)

        for pos_param in list(pos_params.keys()):
            if pos_param not in self._param_root_names:
                self._param_root_names.append(pos_param)

        self._def_values = pos_params

    def _make_param_names(self):
        self._make_param_root_names()
        self.param_names = [self._prefix + name for name in self._param_root_names]
        self._root2full = dict(zip(self._param_root_names, self.param_names))

    def _make_parameters(self, **kwargs) -> dict[str, Fitparam]:
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
        raise NotImplementedError("get_extrema not implemented for this model")

    def get_period(self, params: dict) -> float:
        raise NotImplementedError("get_period not implemented for this model")

    def __add__(self, other):
        return SumModel(self, other)


####################################################################################################
#                   ABC Model Class __add__ function                                               #
####################################################################################################
class SumModel(ModelABC):
    def __init__(self, *models):
        self.models = [copy.deepcopy(model) for model in models]
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

    def _set_param_root_name_and_get_prefix(self):
        root_names_temp = [model._param_root_names for model in self.models]

        for i, param_list in enumerate(root_names_temp):
            if common_member(*root_names_temp):
                self._postfix[i] = f"_{i}"
                param_list = [f"{name}_{i}" for name in param_list]
            self._param_root_names += param_list

            limit = len(param_list)
            if hasattr(self.models[i], "poly_degree"):
                limit = self.models[i].poly_degree + 1

            self.models[i].param_names = param_list[:limit]
            self.param_names += [f"{self._prefix[i]}{name}" for name in param_list][
                :limit
            ]

    def func(self, x, *args, **kwargs):
        if args:
            kwargs = dict(zip(self.param_names, args))

        x = np.array(x)
        func_sum = np.zeros(x.shape)
        for i, model in enumerate(self.models):
            p = self._postfix[i]
            model_kwargs = {
                k.replace(p, ""): v for k, v in kwargs.items() if k in model.param_names
            }
            func_sum += model.func(x, **model_kwargs)
        return func_sum

    def guess(
        self, x: Union[float, Iterable], y: Union[float, Iterable], *args, **kwargs
    ):
        x, y = np.array(x), np.array(y)
        guess = {}
        for model in self.models:
            model_guess = model.guess(x=x, y=y, *args, **kwargs)
            guess |= model_guess

            guess_values = {k: v.values for k, v in model_guess.items()}
            y_guess = model.func(x, **guess_values)
            y = y - y_guess

        return guess

    def funcname(self, *args, **kwargs):
        func_names, func_strs = "", ""
        for model in self.models:
            func_name, func_str = model.funcname(*args, **kwargs).split(" = ")
            func_names += f"{func_name}".replace("(x)", "_")
            func_strs += f"{func_str} + "

        return self._get_function_str(func_names, func_strs)

    def funcname_latex(self, *params) -> str:
        func_names, func_strs = "", ""
        for model in self.models:
            func_name, func_str = model.funcname_latex(*params).split(" = ")
            func_names += f"{func_name}".replace("(x)", "_")
            func_strs += f"{func_str} + "

        return self._get_function_str(func_names, func_strs)

    def _get_function_str(self, func_names, func_strs):
        func_names = "(x)".join(func_names.rsplit("_", 1))
        func_strs = "".join(func_strs.rsplit(" + ", 1))
        return f"{func_names} = {func_strs}"

    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        units = {}
        for model in self.models:
            units |= model.units(x=x, y=y)
        return units

    def get_extrema(self, params: dict) -> dict[str, dict[str, float]]:
        extrema = {}
        for i, model in enumerate(self.models):
            p = self._postfix[i]
            model_params = {
                k.replace(p, ""): v for k, v in params.items() if k in model.param_names
            }
            model_name = f"{model.__class__.__name__}{p}"
            extrema |= {model_name: model.get_extrema(model_params)}
        return extrema


####################################################################################################
#                   Linear Model                                                                   #
####################################################################################################
class LinearModel(ModelABC):
    def __init__(self, independent_vars=None, prefix="", **kwargs):
        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
        super().__init__(**kwargs)

    def func(self, x, slope=1.0, intercept=0.0):
        x = np.array(x)
        return slope * x + intercept

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        slope, intercept = np.polyfit(x, y, 1)
        return self._make_parameters(slope=slope, intercept=intercept)

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_names

        return f"linear(x) = {params[0]} x + {params[1]}"

    def funcname_latex(self, *params) -> str:
        if not params:
            params = self.param_names
        return f"\\mathrm{{linear}}(x) = {params[0]} x + {params[1]}"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"slope": "y/x", "intercept": "y"}

    def get_extrema(self, params: dict) -> dict[str, float]:
        raise ValueError("Linear model has no extrema")


####################################################################################################
#                   Proportional Model                                                                   #
####################################################################################################
class ProportionalModel(ModelABC):
    def __init__(self, independent_vars=None, prefix="", **kwargs):
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
            params = self.param_names

        return f"proportional(x) = {params[0]}*x "

    def funcname_latex(self, *params) -> str:
        if not params:
            params = self.param_names
        return f"\\mathrm{{proportional}}(x) = {params[0]}* x"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"slope": "y/x"}

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
            params = self.param_names
        return f"gaussian(x) = {params[0]} exp(-((x - {params[1]}) / {params[2]})^2)"

    def funcname_latex(self, *params) -> str:
        if not params:
            params = self.param_names
        return f"\\mathrm{{gaussian}}(x) = {params[0]} \\exp(-((x - {params[1]}) / {params[2]})^2)"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "center": "x", "sigma": "x"}

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
            params = self.param_names
        return f"gaussian(x) = {params[0]} exp(-((x - {params[1]}) / {params[2]})^2 + {params[3]})"

    def funcname_latex(self, *params) -> str:
        if not params:
            params = self.param_names
        return f"\\mathrm{{gaussian}}(x) = {params[0]} \\exp(-((x - {params[1]}) / {params[2]})^2) + {params[3]}"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "center": "x", "sigma": "x", "offset": "y"}

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
            params = self.param_names

        return f"lorentzian(x) = {params[0]} (sigma/2)^2 / ((x - {params[1]})^2 + (sigma/2)^2)"

    def funcname_latex(self, *params) -> str:
        if not params:
            params = self.param_names
        return f"\\mathrm{{lorentzian}}(x) = {params[0]} (\\sigma/2)^2 / ((x - {params[1]})^2 + (\\sigma/2)^2)"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "center": "x", "sigma": "x"}

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
            params = self.param_names

        return f"lorentzian(x) = {params[0]} (sigma/2)^2 / ((x - {params[1]})^2 + (sigma/2)^2) + {params[3]}"

    def funcname_latex(self, *params) -> str:
        if not params:
            params = self.param_names
        return f"\\mathrm{{lorentzian}}(x) = {params[0]} (\\sigma/2)^2 / ((x - {params[1]})^2 + (\\sigma/2)^2) + {params[3]}"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "center": "x", "sigma": "x", "offset": "y"}

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

    def funcname_latex(self, *params) -> str:
        return self.__model.funcname_latex(*params)

    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return self.__model.units(x, y)


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
            params = self.param_names

        return (
            f"split_lorentzian(x) = {params[0]} (sigma/2)^2 / ((x - {params[1]} - {params[3]})^2 + (sigma/2)^2) + "
            f"{params[0]} (sigma/2)^2 / ((x - {params[1]} + {params[3]})^2 + (sigma/2)^2)"
        )

    def funcname_latex(self, *params) -> str:
        if not params:
            params = self.param_names
        return (
            f"\\mathrm{{split_lorentzian}}(x) = {params[0]} (\\sigma/2)^2 / ((x - {params[1]} - {params[3]})^2 + (\\sigma/2)^2) + "
            f"{params[0]} (\\sigma/2)^2 / ((x - {params[1]} + {params[3]})^2 + (\\sigma/2)^2)"
        )

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "center": "x", "sigma": "x", "split": "x"}

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
            params = self.param_names
        return f"polynomial(x) = {' + '.join(f'{params[i]} x^{i}' for i in range(self.poly_degree +1))}"

    def funcname_latex(self, *params) -> str:
        if not params:
            params = self.param_names
        return f'\\mathrm{{polynomial}}(x) = {" + ".join(f"{params[i]} x^{i}" for i in range(self.poly_degree +1))}'

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        coeffs_units = {"c0": "y"}
        for i in range(1, self.poly_degree + 1):
            coeffs_units[f"c{i}"] = f"y / x**{i}"

        return coeffs_units

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
#                   Oscillation Model                                                       #
####################################################################################################
class OscillationModel(ModelABC):
    def __init__(self, independent_vars=None, prefix="", **kwargs):
        self.angular = kwargs.pop("angular", False)
        kwargs |= {"prefix": prefix, "independent_vars": independent_vars or ["x"]}
        super().__init__(**kwargs)

    def func(self, x, A=1.0, f=0.0, φ=0.0, c=0.0):
        x = np.array(x)
        if not self.angular:
            f = f * 2 * np.pi
        return A * np.sin(f * x + φ) + c

    def guess(self, x: Union[float, Iterable], y: Union[float, Iterable]) -> dict:
        x, y = np.array(x), np.array(y)
        [amplitude, frequency, phi, offset] = self._oscillations_guess(x, y)

        return self._make_parameters(
            A=amplitude,
            f=frequency,
            φ=phi,
            c=offset,
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
            params = self.param_names
        f = f"{params[1]}" if self.angular else f"2π{params[1]}"
        return f"{params[0]} * sin({f} * x + {params[2]}) + {params[3]}"

    def funcname_latex(self, *params) -> str:
        if not params:
            params = self.param_names
        f = f"{params[1]}" if self.angular else f"2π{params[1]}"
        return f"{params[0]} \\sin({f} x + {params[2]}) + {params[3]}"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {
            "A": "y",
            "f": "x**(-1)",
            "φ": "x",
            "c": "y",
        }

    def get_period(self, params: dict) -> dict[str, float]:
        if self.angular:
            period = 1 / (2 * np.pi * params["f"]["value"])
            error = params["f"]["error"] / params["f"]["value"] ** 2 / (2 * np.pi)
        else:
            period = 1 / params["f"]["value"]
            error = params["f"]["error"] / params["f"]["value"] ** 2

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
        p = 0
        return [a, T, f, p, c]

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_names
        f = f"{params[1]}" if self.angular else f"2π{params[1]}"
        return f"{params[0]} * sin({f} * x + {params[2]}) * exp(-x / {params[3]}) + {params[4]}"

    def funcname_latex(self, *params) -> str:
        if not params:
            params = self.param_names
        f = f"{params[1]}" if self.angular else f"2π{params[1]}"
        return f"{params[0]} \\sin({f} x + {params[2]}) \\exp(-x / {params[3]}) + {params[4]}"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {
            "amplitude": "y",
            "frequency": "x**(-1)",
            "phi": "x",
            "decay": "x**(-1)",
            "offset": "y",
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
            params = self.param_names
        return f"{params[0]} * {params[1]} ^ x + {params[2]}"

    def funcname_latex(self, *params) -> str:
        if not params:
            params = self.param_names
        return f"{params[0]} {params[1]} ^ x + {params[2]}"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "phase": "", "offset": "y"}


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
        offset = np.min(y) - 1e-4  # Small error added
        decay = (x[-1] - x[0]) / -np.log((y[-1] - offset) / (y[0] - offset))
        amplitude = (y[0] - offset) / np.exp(-x[0] / decay)
        return self._make_parameters(amplitude=amplitude, decay=decay, offset=offset)

    def funcname(self, *params) -> str:
        if not params:
            params = self.param_names
        return f"decay(x)={params[0]} * exp(-x / {params[1]}) + {params[2]}"

    def funcname_latex(self, *params) -> str:
        if not params:
            params = self.param_names
        return f"decay(x)={params[0]} \\exp(-x / {params[1]}) + {params[2]}"

    @unit_wrapper
    def units(self, x: Union[float, str, None], y: Union[float, str, None]):
        return {"amplitude": "y", "decay": "x**(-1)", "offset": "y"}
