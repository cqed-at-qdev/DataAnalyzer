# Author: Malthe Asmus Marciniak Nielsen
from typing import Union


def unit_wrapper(func):
    def convert_unit_to_str_or_float(
        f: str, x: Union[float, str, None], y: Union[float, str, None]
    ):
        if type(f) != str:
            raise ValueError(f"f must be a string, not {type(f)}: {f}")

        if not x or not y:
            raise ValueError("x and y must be defined")

        if type(x) != type(y):
            raise TypeError("x and y must be of the same type")

        if isinstance(x, str) and isinstance(y, str):
            f = f.replace("x", x).replace("y", y)
            return f.replace("**", "^")

        elif isinstance(x, (int, float)):
            evalue = eval(f)
            if isinstance(evalue, (float, int)):
                return evalue
            else:
                raise ValueError("f must evaluate to a number")

    def make_units(
        self, **kwargs: dict[str, float or str]
    ) -> dict[str, Union[float, str]]:
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

    def wrapper(self, *args, **kwargs) -> dict[str, Union[float, str]]:
        fs = func(self, *args, **kwargs)
        x = args[0] if args else kwargs.get("x", None)
        y = args[1] if len(args) > 1 else kwargs.get("y", None)

        f_dict: dict[str, Union[float, str]] = {
            k: convert_unit_to_str_or_float(f=f, x=x, y=y) for k, f in fs.items()
        }  # type: ignore
        return make_units(self, **f_dict)

    return wrapper