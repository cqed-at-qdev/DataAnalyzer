import Labber
import numpy as np
from dataanalyzer.utilities import load_json_file


def make_labber_dict(json_parameters, json_results):
    def make_logStep(json_parameters):
        logStep = []
        for data in json_parameters:
            data_dict = data.todict()
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

    logStep = make_logStep(json_parameters)
    logLog = make_logLog(json_results)
    return logStep, logLog


def make_inital_Labber_file(json_filename, logLog, logStep):
    # Create a labber file
    tail = json_filename.split("/")[-1].replace(".json", "")
    f = Labber.createLogFile_ForData(tail, logLog, logStep)

    print([log["name"] for log in logStep])
    for i in range(len(logStep[-1]["values"])):

        data = {Log["name"]: np.array(Log["values"])[i].T for Log in logLog}
        f.addEntry(data)

    # for i in range(step_dimensions[0]):
    #     print(i)
    #     data = {Log["name"]: np.array(Log["values"][i]) for Log in logLog}
    #     f.addEntry(data)

    # for log in logLog:
    #     f.addEntry({log["name"]: np.array(log["values"])})
    #     # for value in log["values"]:
    # f.addEntry({log["name"]: np.array(value)})

    # # Add the data to the file for each step
    # for index in itertools.product(*[range(i) for i in step_dimensions]):
    #     # convert index to a slice
    #     index = slice(*index)

    #     data = {Log["name"]: np.array(Log["values"][index]) for Log in logLog}
    #     f.addEntry(data)

    return f.getFilePath("")


def make_Labber_file(json_filename, json_parameters, json_results):
    logStep, logLog = make_labber_dict(json_parameters, json_results)
    return make_inital_Labber_file(json_filename, logLog, logStep)


def json2Labber(json_filename):
    json_parameters, json_results = load_json_file(json_filename)
    return make_Labber_file(json_filename, json_parameters, json_results)


if __name__ == "__main__":
    # Load json file
    json_path = r"C:\Users\T5_2\Desktop\quantum machines demo\data20230106\113525_state_after_2D_Rabi_chevron_freq_vs_amplitude.json"

    # Convert json to labber
    labber_path = json2Labber(json_path)

    # Test labber file
    # test_Labber_file(labber_path, json_path)
    print("Labber file saved to: ", labber_path)
