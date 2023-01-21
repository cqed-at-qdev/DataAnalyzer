# Author: Malthe Asmus Marciniak Nielsen

import json
from typing import Optional, Tuple, Union

import numpy as np

from dataanalyzer.utilities.valueclass import Valueclass

try:
    import Labber
except ImportError:
    import dataanalyzer.local_labber as Labber


####################################################################################################
#                   From Valueclass to Dict                                                        #
####################################################################################################
def valueclass2dict(
    *value: Union[Valueclass, list, tuple, dict], split_complex=True
) -> dict:
    """Converts a Valueclass object to a dict.

    Args:
        value (Valueclass): The value to be converted.

    Returns:
        dict: The converted value.
    """
    # Create an empty dictionary
    valuedict = {}

    # Loop over all values
    for v in value:
        # If the value is a Valueclass object, convert it to a dict and add it to the dictionary
        if isinstance(v, Valueclass):
            valuedict[v.name] = v.todict(split_complex=split_complex)

        # If the value is a list or tuple, convert each element to a dict and add it to the dictionary
        elif isinstance(v, (list, tuple)):
            valuedict |= _valueclass2dict_list(v)

        # If the value is a dict, convert each element to a dict and add it to the dictionary
        elif isinstance(v, dict):
            valuedict |= _valueclass2dict_dict(v)

        # If the value is a int, float, str, bool, add it to the dictionary
        elif isinstance(v, (int, float, str, bool)):
            valuedict = v

        # If the value is not of one of the above types, raise an error
        else:
            raise TypeError(f"Valueclass2dict does not accept type {type(v)}.")
    return valuedict


def _valueclass2dict_list(value: Union[list, tuple]) -> dict:
    """Converts a list to a dict.

    Args:
        value (list): The list to be converted.

    Returns:
        dict: The converted list.
    """
    valuedict = {}
    for vi in value:
        valuedict |= valueclass2dict(vi)
    return valuedict


def _valueclass2dict_dict(value: dict) -> dict:
    """Converts a dict to a dict.

    Args:
        value (dict): The dict to be converted.

    Returns:
        dict: The converted dict.
    """
    valuedict = {}
    if "name" in value and "value" in value:
        v_valueclass = Valueclass.fromdict(value)
        valuedict[v_valueclass.name] = v_valueclass
    else:
        for ki, vi in value.items():
            valuedict[ki] = valueclass2dict(vi)
    return valuedict


####################################################################################################
#                   From Dict to Valueclass                                                        #
####################################################################################################
def dict2valueclass(*value: dict) -> list[Valueclass]:
    """Converts a dict to a Valueclass object.

    Args:
        value (dict): The value to be converted.

    Returns:
        Valueclass: The converted value.
    """
    valuelist = []

    # Loop through each value
    for v in value:
        # If v is a dict
        if isinstance(v, dict):
            # Check if the dict has the keys "name" and "value"
            if "name" in v and "value" in v:
                # If so, convert the dict to a Valueclass object
                valuelist.append(Valueclass.fromdict(v))
            # If not, check if the dict contains other dicts
            else:
                # If so, convert each dict to a Valueclass object
                valuelist.extend(dict2valueclass(vi) for vi in v.values())

    return valuelist


####################################################################################################
#                   From Valueclass to Json                                                        #
####################################################################################################
def valueclass2json(
    parameters: Union[Valueclass, list, tuple, dict],
    results: Union[Valueclass, list, tuple, dict],
    path: str,
) -> None:
    """Converts a Valueclass object to a json string.

    Args:
        parameters (Union[Valueclass, list, tuple, dict]): The parameters to be converted.
        results (Union[Valueclass, list, tuple, dict]): The results to be converted.
        path (str): The path to the json file.
    """

    # Convert the parameters and results to a dictionary.
    settings_dict = valueclass2dict(parameters, split_complex=True)
    result_dict = valueclass2dict(results, split_complex=True)

    # Create a dictionary with the experiment settings and results.
    jsondict = {"experiment_settings": settings_dict, "experiment_results": result_dict}

    # Add the dictionary to the json file.
    with open(path, "r+") as file:
        try:
            json_dict = json.load(file)
        except json.decoder.JSONDecodeError:
            json_dict = {}
        json_dict |= jsondict
        file.seek(0)
        json.dump(json_dict, file, indent=4)


####################################################################################################
#                   From Json to Valueclass                                                        #
####################################################################################################
def json2valueclass(
    json_path: str, insepct: bool = False
) -> Tuple[list[Valueclass], list[Valueclass]]:
    """Converts a json file to a Valueclass object.

    Args:
        json_path (str): The path to the json file.
        insepct (bool, optional): If True, the parameters and results are printed. Defaults to False.

    Returns:
        Tuple[list[Valueclass], list[Valueclass]]: _description_
    """

    # Open the json file
    with open(json_path, "r") as f:
        data_dict = json.load(f)

    # Get the parameters and results
    parameters, results = _get_parameters_and_results_from_dict(data_dict)

    # Inspect the results if insepct is True
    _print_parameters_and_results(
        parameters, results, insepct=insepct, file_type="Json"
    )
    return parameters, results


def _get_parameters_and_results_from_dict(
    data_dict: dict,
) -> Tuple[list[Valueclass], list[Valueclass]]:
    """Get the parameters and results from the json file.

    Args:
        data_dict (dict): The dictionary containing the parameters and results.

    Returns:
        Tuple[list[Valueclass], list[Valueclass]]: The parameters and results.
    """
    # Get the parameters from the data_dict
    parameters = [
        Valueclass.fromdict(value)
        for value in data_dict["experiment_settings"].values()
    ]
    # Get the results from the data_dict
    results = [
        Valueclass.fromdict(value) for value in data_dict["experiment_results"].values()
    ]
    return parameters, results


####################################################################################################
#                   From Valueclass to Labber                                                      #
####################################################################################################
def valueclass2labber(
    parameters: list[Valueclass], results: list[Valueclass], output_path: str
) -> str:
    """Converts a Valueclass object to a Labber file.

    Args:
        parameters (list[Valueclass]): Valueclass parameters.
        results (list[Valueclass]): Valueclass results.
        output_path (str): The path to the Labber file.

    Returns:
        str: The path to the Labber file.
    """
    # Create the Labber dictionaries
    logStep, logLog = _make_labber_dict(parameters, results)

    # Create the Labber file and return the path
    return _make_inital_Labber_file(output_path, logLog, logStep)


def _make_labber_dict(
    parameters: list[Valueclass], results: list[Valueclass]
) -> Tuple[list[dict], list[dict]]:
    """Make Labber logStep and logLog dictionaries.

    Args:
        parameters (list[Valueclass]): Labber parameters.
        results (list[Valueclass]): Labber results.

    Returns:
        Tuple[list[dict], list[dict]]: Labber logStep and logLog dictionaries.
    """
    # Create a list of dictionaries for the step data
    logStep = [
        dict(name=data.name, unit=data.unit, values=data.value.tolist())
        for data in parameters
    ]

    # Create a list of dictionaries for the log data
    logLog = [
        dict(
            name=data.name,
            unit=data.unit,
            values=data.value.tolist(),
            vector=False,
            complex=True,
        )
        for data in results
    ]

    # Return the step and log data
    return logStep, logLog


def _make_inital_Labber_file(path: str, logLog: list[dict], logStep: list[dict]) -> str:
    """Make the initial Labber file.

    Args:
        path (str): The path to the Labber file.
        logLog (list[dict]): List of dictionaries containing the log data.
        logStep (list[dict]): List of dictionaries containing the step data.

    Returns:
        str: The path to the Labber file.
    """
    # Create the Labber file
    f = Labber.createLogFile_ForData(path, logLog, logStep)

    # Add the data to the Labber file for each step
    for i in range(len(logStep[-1]["values"])):
        # Create a dictionary with the data
        data = {Log["name"]: np.array(Log["values"])[i].T for Log in logLog}
        f.addEntry(data)

    # Close the Labber file and return the path
    return f.getFilePath("")


####################################################################################################
#                   From Labber to Valueclass                                                      #
####################################################################################################
def labber2valueclass(
    labber_path: str, insepct: bool = False
) -> Tuple[list[Valueclass], list[Valueclass]]:
    """Converts a Labber file to a Valueclass object.

    Args:
        labber_path (str): The path to the Labber file.
        insepct (bool, optional): If True, the parameters and results are printed. Defaults to False.

    Returns:
        Tuple[list[Valueclass], list[Valueclass]]: The parameters and results.
    """
    # Open Labber file
    f = Labber.LogFile(labber_path)

    # Get parameters and results
    parameters = _get_parameters(f)
    results = _get_results(f)

    # Inspect the results if insepct is True
    _print_parameters_and_results(
        parameters, results, insepct=insepct, file_type="Labber"
    )

    return parameters, results


def _get_vna_data(step_channels, channels) -> list[Valueclass]:
    """Calculate the frequencies for the VNA measurement.

    The VNA measurement can be performed in three ways:
    - Specify start and stop frequency
    - Specify center frequency and span
    - Specify the number of points (the start and stop frequency will be calculated)

    Args:
        step_channels (list): The channels in the step
        channels (dict): The channels in the experiment

    Returns:
        list: A list of Valueclass objects
    """
    start_freq = stop_freq = center_freq = span = n_points = None

    # Find the start and stop frequency
    for param in step_channels:
        if "VNA" in param["name"]:
            if "Start frequency" in param:
                start_freq = param["values"]
            elif "Stop frequency" in param:
                stop_freq = param["values"]

    # Find the number of points, center frequency and span
    for param, value in channels.items():
        if "VNA" in param:
            if "# of points" in param:
                n_points = value

            elif start_freq is None or stop_freq is None:
                if "Center frequency" in param:
                    center_freq = value
                elif "Span" in param:
                    span = value

    # Calculate the start and stop frequency if center frequency and span is specified
    if center_freq is not None and span is not None:
        start_freq = center_freq - span / 2
        stop_freq = center_freq + span / 2

    # Calculate the frequencies if start and stop frequency is specified
    if start_freq is not None and stop_freq is not None and n_points is not None:
        values = np.linspace(float(start_freq), float(stop_freq), int(n_points))
        return [
            Valueclass(
                name="VNA - Frequency",
                value=values,
                unit="Hz",
            )
        ]
    return []


def _get_parameters(logfile: Labber.LogFile) -> list[Valueclass]:
    """Get the parameters from the Labber file.

    Args:
        logfile (Labber.LogFile): The Labber file.

    Returns:
        list[Valueclass]: A list of Valueclass objects.
    """
    # Get step channels and channel values as a dictionary
    step_channels = logfile.getStepChannels()
    channels = logfile.getChannelValuesAsDict()

    # Get the VNA data
    parameters = _get_vna_data(step_channels, channels)

    # Add all step channels that have more than one value
    parameters += [
        Valueclass(name=step["name"], value=step["values"], unit=step["unit"])
        for step in step_channels
        if len(step["values"]) > 1
    ]
    return parameters


def _get_results(logfile: Labber.LogFile) -> list[Valueclass]:
    """Get the results from the Labber file.

    Args:
        logfile (Labber.LogFile): The Labber file.

    Returns:
        list[Valueclass]: A list of Valueclass objects.
    """
    results = []
    log_channels = logfile.getLogChannels()
    for i, log_channel in enumerate(log_channels):
        # Get the data from the Labber file
        data = logfile.getData(i)
        # Create a Valueclass object for each log channel
        results.append(
            Valueclass(
                name=log_channel["name"],
                value=data[0] if len(data) < 2 else data,
                unit=log_channel["unit"],
            )
        )
    return results


####################################################################################################
#                   From Json to Labber                                                            #
####################################################################################################
def json2labber(json_path: str, output_path: Optional[str] = None) -> str:
    """Converts a Json file to a Labber file.

    Args:
        json_path (str): The path to the Json file.
        output_path (Optional[str], optional): The path to the Labber file.
        If None, the path is set to the same as the Json file. Defaults to None.

    Returns:
        str: The path to the Labber file.
    """
    # If the output path is not specified, set it to the same as the Json file
    if output_path is None:
        output_path = json_path.replace(".json", ".hdf5")

    # Get the parameters and result from the Json file
    parameters, result = json2valueclass(json_path)

    # Convert the parameters and result to a Labber file
    return valueclass2labber(parameters, result, output_path)


####################################################################################################
#                   From Labber to Json                                                            #
####################################################################################################
def labber2json(labber_path: str, output_path: Optional[str] = None) -> str:
    """Converts a Labber file to a Json file.

    Args:
        labber_path (str): The path to the Labber file.
        output_path (Optional[str], optional): The path to the Json file. If None, the path is set to the same as the Labber file. Defaults to None.

    Returns:
        str: The path to the Json file.
    """
    # Set the output path if not given
    if output_path is None:
        output_path = labber_path.replace(".hdf5", ".json")

    # Get the data and parameters
    parameters, result = labber2valueclass(labber_path)

    # Convert the data and parameters to a Json file
    valueclass2json(parameters, result, output_path)
    return output_path


####################################################################################################
#                   Other functions                                                                #
####################################################################################################
def _print_parameters_and_results(
    parameters: list[Valueclass],
    results: list[Valueclass],
    insepct: bool,
    file_type: str,
) -> None:
    """Prints the parameters and results to the console.

    Args:
        parameters (list[Valueclass]): list of Valueclass objects containing the parameters
        results (list[Valueclass]): list of Valueclass objects containing the results
        insepct (bool): If True, the parameters and results are printed to the console
        file_type (str): The type of file, e.g. Labber or Json
    """
    if insepct:
        print(f"Insepcting {file_type} File...")
        print_values_from_list("parameters", parameters)
        print_values_from_list("results", results)


def print_values_from_list(name: str, value_list: list[Valueclass]) -> None:
    """Prints the values from a list of Valueclass objects to the console.

    Args:
        name (str): The name of the list of Valueclass objects (e.g. parameters or results)
        value_list (list[Valueclass]): The list of Valueclass objects
    """
    print(f"File is containing {len(value_list)} {name}:")
    for value in value_list:
        print(f"\t{value.name}")


if __name__ == "__main__":

    class test_conversion:
        def __init__(self) -> None:
            self.v1 = Valueclass(name="test1", value=1, unit="V")
            self.v2 = Valueclass(name="test2", value=[1, 2, 3], unit="V")
            self.v3 = Valueclass(name="test3", value=[1 + 1j, 2 + 2j, 3 + 3j], unit="V")

            self.d1 = {"name": "test1", "value": 1, "unit": "V"}

        def run_tests(self):
            self.test_valueclass2dict()
            self.test_dict2valueclass()

        def test_valueclass2dict(self):
            assert valueclass2dict(self.v1) == {self.v1.name: self.v1.todict()}
            assert valueclass2dict(self.v2) == {self.v2.name: self.v2.todict()}

            assert valueclass2dict([self.v1, self.v2]) == {
                self.v1.name: self.v1.todict(),
                self.v2.name: self.v2.todict(),
            }
            assert valueclass2dict({"test": [self.v1, self.v2]}) == {
                "test": {self.v1.name: self.v1.todict(), self.v2.name: self.v2.todict()}
            }
            assert valueclass2dict({"test": self.v1}, self.v2) == {
                "test": {self.v1.name: self.v1.todict()},
                self.v2.name: self.v2.todict(),
            }

        def test_dict2valueclass(self):
            d1_value = dict2valueclass(self.d1)[0]
            assert d1_value.name == self.d1["name"]
            assert d1_value.value == np.array(self.d1["value"])
            assert d1_value.unit == self.d1["unit"]

        def test_valueclass2json(self):
            pass

        def test_valueclass2labber(self):
            pass

    test_conversion().run_tests()
