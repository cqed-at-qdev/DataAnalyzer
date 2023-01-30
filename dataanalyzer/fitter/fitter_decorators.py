# Author: Malthe Asmus Marciniak Nielsen
from typing import Union


def unit_wrapper(func):
    def convert_unit_to_str_or_float(
        f: str, x: Union[float, str, None], y: Union[float, str, None]
    ):
        if type(f) != str:
            raise ValueError(f"f must be a string, not {type(f)}: {f}")

        if x is None or y is None:
            raise ValueError("x and y must be defined")

        if type(x) != type(y):
            raise TypeError("x and y must be of the same type")

        if isinstance(x, str) and isinstance(y, str):
            f = f.replace("x", x).replace("y", y)
            return f.replace("**", "^").replace("(", "{").replace(")", "}")

        elif isinstance(x, (int, float)):
            try:
                evalue = eval(f)
            except Exception as e:
                return 1

            if isinstance(evalue, (float, int)):
                return evalue
            else:
                raise ValueError("f must evaluate to a number")

    def update_parameters(self, **kwargs: dict[str, float or str]):
        for i, parameter in enumerate(self.parameters):
            if parameter.base_name in kwargs:
                self.parameters[i].unit = kwargs[parameter.base_name]

    # def make_units(
    #     self, **kwargs: dict[str, float or str]
    # ) -> dict[str, Union[float, str]]:
    #     units = {}
    #     for parameter in self.parameters:

    #         # basename = name[len(self._prefix) :]

    #         # # Updates unit from kwargs (no prefix)
    #         # if basename in kwargs:
    #         #     units[name] = kwargs[basename]

    #         # # Updates unit from kwargs (with prefix)
    #         # if name in kwargs:
    #         #     units[name] = kwargs[name]

    #         # if (
    #         #     self._frequency_in_hz
    #         #     and "frequency" in name
    #         #     and isinstance(units[name], str)
    #         # ):
    #         #     units[name] = units[name].replace("s^{-1}", "Hz")

    #     return units

    def wrapper(self, *args, **kwargs) -> dict[str, Union[float, str]]:
        unit_dict = func(self, *args, **kwargs)
        update_parameters(self, **unit_dict)

        x = args[0] if args else kwargs.get("x", None)
        y = args[1] if len(args) > 1 else kwargs.get("y", None)

        f_dict: dict[str, Union[float, str]] = {
            parameter.full_name: convert_unit_to_str_or_float(
                f=parameter.unit, x=x, y=y
            )
            for parameter in self.parameters
        }  # type: ignore
        return f_dict

    return wrapper


def symbol_wrapper(func):
    def wrapper(self, *args, **kwargs) -> None:
        symbol_dict = func(self, *args, **kwargs)

        for key, value in self.parameters.items():
            if key in symbol_dict:
                value.symbol = symbol_dict[key]

    return wrapper


if __name__ == "__main__":
    from dataanalyzer.fitter.fitter_classsetup import Fitparam

    class DommyModel:
        def __init__(self):
            self._prefix = "prefix_"
            self._suffix = "_suffix"
            self.parameters = [
                Fitparam(base_name="a", model=self),
                Fitparam(base_name="b", model=self),
                Fitparam(base_name="c", model=self),
            ]

        @unit_wrapper
        def get_units(self, x, y):
            return {"a": "x", "b": "y", "c": "x**2"}

    model = DommyModel()
    print(model.get_units("test", "something"))
    print(model.parameters[0])
