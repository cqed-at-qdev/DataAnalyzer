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

    def _get_parameters(f):
        step_channels = f.getStepChannels()

        parameters = _get_vna_data(step_channels)
        parameters += [
            Valueclass(name=step["name"], value=step["values"], unit=step["unit"])
            for step in step_channels
            if len(step["values"]) > 1
        ]
        return parameters

    def _get_data(f):
        log_channel = f.getLogChannels()[0]
        d = f.getData()
        data = Valueclass(
            name=log_channel["name"],
            value=d[0] if len(d) < 2 else d,
            unit=log_channel["unit"],
        )
        return [data]

    try:
        import Labber
    except ImportError:
        import dataanalyzer.local_labber as Labber

    f = Labber.LogFile(labber_path)

    parameters = _get_parameters(f)
    data = _get_data(f)

    if insepct:
        print("Insepcting Json File...")
        print_values_from_list("parameters", parameters)
        print_values_from_list("results", data)

    return *parameters, *data


def load_json_file(
    json_path: str, insepct: bool = False
) -> Tuple[list[Valueclass], list[Valueclass]]:
    def _get_parameters_and_results(data_dict):
        parameters = [
            Valueclass.fromdict(value)
            for value in data_dict["experiment_settings"].values()
        ]
        result = [
            Valueclass.fromdict(value)
            for value in data_dict["experiment_results"].values()
        ]
        return parameters, result

    import json

    with open(json_path, "r") as f:
        data_dict = json.load(f)

    parameters, result = _get_parameters_and_results(data_dict)

    if insepct:
        print("Insepcting Json File...")
        print_values_from_list("parameters", parameters)
        print_values_from_list("results", result)

    return parameters, result


def print_values_from_list(name: str, value_list: list[Valueclass]):
    print(f"File is containing {len(value_list)} {name}:")
    for value in value_list:
        print(f"\t{value.name}")


if __name__ == "__main__":
    fpath = r"C:\Users\T5_2\Desktop\quantum machines demo\data20230105\152337_state_after_protective_freq_vs_theta.json"
    load_json_file(fpath, insepct=True)
