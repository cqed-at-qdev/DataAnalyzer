# Author: Malthe Asmus Marciniak Nielsen
import json
from typing import Optional, Tuple, Union

import Labber
import numpy as np

from dataanalyzer.utilities.valueclass import Valueclass


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
    valuedict = {}
    for v in value:
        if isinstance(v, Valueclass):
            valuedict[v.name] = v.todict(split_complex=split_complex)

        elif isinstance(v, (list, tuple)):
            for vi in v:
                valuedict |= valueclass2dict(vi)

        elif isinstance(v, dict):
            if "name" in v and "value" in v:
                v_valueclass = Valueclass.fromdict(v)
                valuedict[v_valueclass.name] = v_valueclass
            else:
                for ki, vi in v.items():
                    valuedict.update({ki: valueclass2dict(vi)})
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

    for v in value:
        if isinstance(v, dict):
            if "name" in v and "value" in v:
                valuelist.append(Valueclass.fromdict(v))
            else:
                valuelist.extend(dict2valueclass(vi) for vi in v.values())

    return valuelist


####################################################################################################
#                   From Valueclass to Json                                                        #
####################################################################################################
def valueclass2json(
    exp_settings: Union[Valueclass, list, tuple, dict],
    exp_results: Union[Valueclass, list, tuple, dict],
    path: str,
):
    """Converts a Valueclass object to a json string.
    
    Args:
        valuedict (dict[str, Valueclass]): The value to be converted.
        path (str): The path to the json file.
    """

    def update_json_dict(path, jsondict):
        with open(path, "r+") as file:
            json_dict = json.load(file)
            json_dict.update(jsondict)
            file.seek(0)
            json.dump(json_dict, file, indent=4)

    settings_dict = valueclass2dict(exp_settings, split_complex=True)
    result_dict = valueclass2dict(exp_results, split_complex=True)
    jsondict = {"experiment_settings": settings_dict, "experiment_results": result_dict}

    update_json_dict(path, jsondict)


####################################################################################################
#                   From Json to Valueclass                                                        #
####################################################################################################
def json2valueclass(
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

    with open(json_path, "r") as f:
        data_dict = json.load(f)

    parameters, result = _get_parameters_and_results(data_dict)

    if insepct:
        print("Insepcting Json File...")
        print_values_from_list("parameters", parameters)
        print_values_from_list("results", result)

    return parameters, result


####################################################################################################
#                   From Valueclass to Labber                                                      #
####################################################################################################
def valueclass2labber(parameters: list[Valueclass], results: list[Valueclass], output_path: str):
    def _make_labber_dict(parameters, results):
        def make_logStep(parameters):
            logStep = []
            for data in parameters:
                data_dict = data.todict()
                lab_dict = dict(
                    name=data_dict["name"],
                    unit=data_dict["unit"],
                    values=data_dict["value"],
                )
                logStep.append(lab_dict)
            return logStep

        def make_logLog(results):
            logLog = []
            for data in results:
                data_dict = data.todict()
                lab_dict = dict(
                    name=data_dict["name"],
                    unit=data_dict["unit"],
                    values=data_dict["value"],
                    vector=False,
                    complex=True,
                )
                logLog.append(lab_dict)
            return logLog

        logStep = make_logStep(parameters)
        logLog = make_logLog(results)
        return logStep, logLog

    def _make_inital_Labber_file(path, logLog, logStep):
        f = Labber.createLogFile_ForData(path, logLog, logStep)
        
        for i in range(len(logStep[-1]["values"])):
            data = {Log["name"]: np.array(Log["values"])[i].T for Log in logLog}
            f.addEntry(data)
            
        return f.getFilePath("")
    
    logStep, logLog = _make_labber_dict(parameters, results)
    return _make_inital_Labber_file(output_path, logLog, logStep)


####################################################################################################
#                   From Labber to Valueclass                                                      #
####################################################################################################
def labber2valueclass(labber_path: str, insepct: bool = False) -> Tuple[Valueclass, ...]:
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


####################################################################################################
#                   From Json to Labber                                                            #
####################################################################################################
def json2labber(json_path: str, output_path: Optional[str] = None):
    if output_path is None:
        output_path = json_path.replace(".json", ".hdf5")
        
    parameters, result = json2valueclass(json_path)
    return valueclass2labber(parameters, result, output_path)


####################################################################################################
#                   From Labber to Json                                                            #
####################################################################################################
def labber2json(labber_path: str, output_path: Optional[str] = None):
    if output_path is None:
        output_path = labber_path.replace(".hdf5", ".json")
        
    parameters, result = labber2valueclass(labber_path)
    return valueclass2json(parameters, result, output_path)


####################################################################################################
#                   Other functions                                                                #
####################################################################################################
def print_values_from_list(name: str, value_list: list[Valueclass]):
    print(f"File is containing {len(value_list)} {name}:")
    for value in value_list:
        print(f"\t{value.name}")


if __name__ == "__main__":
    class test_conversion():
        def __init__(self) -> None:
            self.v1 = Valueclass(name="test1", value=1, unit="V")
            self.v2 = Valueclass(name="test2", value=[1, 2, 3], unit="V")
            self.v3 = Valueclass(name="test3", value=[1+1j, 2+2j, 3+3j], unit="V")
            
            self.d1 = {"name": "test1", "value": 1, "unit": "V"}
            
        def run_tests(self):
            self.test_valueclass2dict()
            self.test_dict2valueclass()
            
        def test_valueclass2dict(self):
            assert valueclass2dict(self.v1) == {self.v1.name: self.v1.todict()}
            assert valueclass2dict(self.v2) == {self.v2.name: self.v2.todict()}
            
            assert valueclass2dict([self.v1, self.v2]) == {self.v1.name: self.v1.todict(), self.v2.name: self.v2.todict()}
            assert valueclass2dict({"test":[self.v1, self.v2]}) == {"test": {self.v1.name: self.v1.todict(), self.v2.name: self.v2.todict()}}
            assert valueclass2dict({"test":self.v1}, self.v2) == {"test": {self.v1.name: self.v1.todict()}, self.v2.name: self.v2.todict()}
        
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