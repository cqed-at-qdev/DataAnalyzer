import Labber
import numpy as np
import h5py
from dataanalyzer import load_json_file
import json
import hdfdict


def make_labber_dict(json_parameters, json_results):
    def make_logStep(json_parameters):
        logStep = []
        for data in json_parameters:
            data_dict = data.asdict()
            lab_dict = dict(
                name=data_dict["name"],
                unit=data_dict["unit"],
                values=data_dict["value"],
            )
            logStep.append(lab_dict)
        return logStep

    def make_logLog(json_results):
        logLog = []
        for data in json_results:
            data_dict = data.asdict()
            lab_dict = dict(
                name=data_dict["name"],
                unit=data_dict["unit"],
                values=data_dict["value"],
                vector=True,
                complex=True,
            )
            logLog.append(lab_dict)
        return logLog

    logStep = make_logStep(json_parameters)
    logLog = make_logLog(json_results)
    return logStep, logLog


def make_inital_Labber_file(json_filename, logLog, logStep):
    # Create a labber file
    tail = json_filename.split("/")[-1].replace(".json", "")
    f = Labber.createLogFile_ForData(tail, logLog, logStep)
    return f.getFilePath("")


def add_data_to_Labber_file(filepath, logLog):
    with h5py.File(filepath, "r+") as f:
        # Add the data to the dictionary
        traces = {"Traces": {log["name"]: np.array(log["values"]) for log in logLog}}

        # Save the file
        hdfdict.dump(traces, f, mode="w")


def make_Labber_file(json_filename, json_parameters, json_results):
    logStep, logLog = make_labber_dict(json_parameters, json_results)
    filepath = make_inital_Labber_file(json_filename, logLog, logStep)
    add_data_to_Labber_file(filepath, logLog)
    return filepath


def test_Labber_file(labber_path, json_path):
    json_parameters, json_results = load_json_file(json_path)
    logStep, logLog = make_labber_dict(json_parameters, json_results)
    with h5py.File(labber_path) as f:
        labber_dict = hdfdict.load(f)
        print("Labber traces:\n", labber_dict["Traces"], "\n")  # type: ignore

    print(
        "Json traces:\n", {log["name"]: np.array(log["values"]) for log in logLog}, "\n"
    )


def json2Labber(json_filename):
    json_parameters, json_results = load_json_file(json_filename)
    return make_Labber_file(json_filename, json_parameters, json_results)


if __name__ == "__main__":
    # Create dommy json file with experiment_settings and experiment_results
    dummy_dict = {
        "experiment_settings": {
            "experiment_settings": {
                "Time": {
                    "name": "Time",
                    "value": [0.0, 0.004, 0.004, 0.006, 0.008, 0.01],
                    "unit": "s",
                },
                "Frequency": {
                    "name": "Frequency",
                    "value": [1.0, 222.0, 3.0, 4.0, 5.0, 6.0],
                    "unit": "Hz",
                },
            },
            "experiment_results": {
                "Signal": {
                    "name": "Signal",
                    "value": [
                        0.0,
                        0.00020000000000000002,
                        0.00040000000000000004,
                        0.0006000000000000001,
                        0.0008000000000000001,
                        0.001,
                    ],
                    "unit": "V",
                }
            },
        }
    }

    json_path = "sample_json/140412_state_after_test_save_data copy.json"
    with open(json_path, "w") as f:
        json.dump(dummy_dict, f, indent=4)

    # Load json file
    labber_path = json2Labber(json_path)
    test_Labber_file(labber_path, json_path)
