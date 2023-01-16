from typing import Union

import numpy as np

from dataanalyzer.utilities.valueclass import Valueclass


def float2valueclass(
    value: Union[Valueclass, list, tuple, np.ndarray], name: str
) -> Valueclass:
    """Converts a float to a Valueclass object.

    Args:
        value (float): The value to be converted.
        name (str): The name of the v.

    Returns:
        Valueclass: The converted value.
    """

    return (
        value if isinstance(value, Valueclass) else Valueclass(name=name, value=value)
    )


def valueclass2dict(*value: Valueclass, complex=False) -> dict:
    """Converts a Valueclass object to a dict.

    Args:
        value (Valueclass): The value to be converted.

    Returns:
        dict: The converted value.
    """
    valuedict = {}
    for v in value:
        if isinstance(v, Valueclass):
            valuedict[v.name] = v.asdict()

        elif isinstance(v, (list, tuple)):
            for vi in v:
                valuedict |= valueclass2dict(vi)

        elif isinstance(v, dict):
            if "name" in v and "value" in v:
                v = Valueclass.fromdict(value)
                valuedict[v.name] = v
            else:
                for vi in v.values():
                    valuedict |= valueclass2dict(vi)
    return valuedict


if __name__ == "__main__":
    # Make a test of valueclass2dict
    test = Valueclass(name="test1", value=1)
    test2 = Valueclass(name="test2", value=2)
    test3 = Valueclass(name="test3", value=3)

    testdict = valueclass2dict(test, (test2, test3), dict(test4=test2, test5=test3))
    print(testdict)


import json


def save_data(filename: str, experiment_settings: list, experiment_results: list):
    data_dict = {
        "experiment_settings": {
            valueclass.name: valueclass.asdict() for valueclass in experiment_settings
        },
        "experiment_results": {
            valueclass.name: valueclass.asdict() for valueclass in experiment_results
        },
    }

    # check if values are complex and if so save them as real and imaginary parts
    for exp in ["experiment_settings", "experiment_results"]:
        for param in list(data_dict[exp].values()):
            if np.iscomplexobj(param["value"]):
                I = np.array(param["value"]).real.tolist()
                Q = np.array(param["value"]).imag.tolist()
                param["value"] = {"I": I, "Q": Q}

    with open(filename, "r+") as file:
        json_dict = json.load(file)
        json_dict.update(data_dict)
        file.seek(0)
        json.dump(json_dict, file, indent=4)
