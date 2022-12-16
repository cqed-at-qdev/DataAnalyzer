# Author: Malthe Asmus Marciniak Nielsen
from typing import Tuple

import numpy as np

from dataanalyzer.utilities.valueclass import Valueclass


def load_labber_file(labber_path: str, insepct: bool = False) -> Tuple[Valueclass, ...]:
    def _get_vna_data(step_channels) -> list[Valueclass]:
        start_freq = stop_freq = n_points = vna_param = None
        for param in step_channels:
            if "VNA" in param["name"]:
                if "Start frequency" in param["name"]:
                    start_freq = param["values"]
                    vna_param = param
                elif "Stop frequency" in param["name"]:
                    stop_freq = param["values"]
                elif "# of points" in param["name"]:
                    n_points = param["values"]
            if (
                start_freq is not None
                and stop_freq is not None
                and n_points is not None
                and vna_param is not None
            ):
                values = np.linspace(float(start_freq), float(stop_freq), int(n_points))
                return [
                    Valueclass(
                        name=vna_param["name"].replace("Start f", "F"),
                        value=values,
                        unit=vna_param["unit"],
                    )
                ]
        return []

    try:
        import Labber
    except ImportError:
        import dataanalyzer.local_labber as Labber

    f = Labber.LogFile(labber_path)
    step_channels = f.getStepChannels()
    log_channel = f.getLogChannels()[0]

    parameters = _get_vna_data(step_channels)
    parameters += [
        Valueclass(name=step["name"], value=step["values"], unit=step["unit"])
        for step in step_channels
        if len(step["values"]) > 1
    ]

    d = f.getData()
    data = Valueclass(
        name=log_channel["name"],
        value=d[0] if len(d) < 2 else d,
        unit=log_channel["unit"],
    )

    if insepct:
        parameter_names = [param.name for param in parameters] + [data.name]
        print(
            f"Insepcting Labber File...\nFile is containing {len(parameter_names)} parameters:"
        )
        for param in parameter_names:
            print(f"\t{param}")
        print("\n")

    return *parameters, data


def load_json_file(json_path: str, insepct: bool = False) -> Tuple[Valueclass, ...]:
    import json

    with open(json_path, "r") as f:
        data_dict = json.load(f)

    parameters = [
        Valueclass.fromdict(value)
        for key, value in data_dict["experiment_data"].items()
    ]

    if insepct:
        parameter_names = [param.name for param in parameters]
        print(
            f"Insepcting Json File...\nFile is containing {len(parameter_names)} parameters:"
        )
        for param in parameter_names:
            print(f"\t{param}")
        print("\n")

    return tuple(parameters)
