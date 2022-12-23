"""
performs qubit spec vs freq and flux to show the parabola
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from macros import *
from qualang_tools.plot import interrupt_on_close, plot_demodulated_data_2d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from dataanalyzer import Valueclass, Plotter
import uuid
from datetime import datetime


import matplotlib

matplotlib.use("QtAgg")
#%matplotlib ipympl


def get_saving_path(filename):
    now = datetime.now()
    now = now.strftime("%m%d%Y_%H%M%S")
    directory = f"{machine.results.directory}{now[:8]}/"
    return f"{directory}{now[9:]}_state_after_{filename}.json"


def live_plot(
    x: Valueclass, y: Valueclass, z: Valueclass, metadata: dict = None, fig=None
):
    if metadata is None:
        metadata = {}
    plot = Plotter(subplots=(2, 1), fig=fig)

    # Amplitude plot (top)
    plot.heatmap(x, y, z.abs, title="Amplitude", cmap="magma")

    # Phase plot (bottom)
    plot.heatmap(x, y, z.phase, title="Phase", cmap="twilight_shifted", ax=(1, 0))

    fit = {
        "function": "gaussian",
        "parameters": "x0 = 0, y0 = 0, sigma_x = 1, sigma_y = 1, amplitude = 1, offset = 0",
    }

    metadata["Fitting report"] = fit
    metadata_str = dict2str(metadata)

    # Add metadata
    plot.add_metadata(metadata_str)

    return plot.show(return_fig=True)


##################
# State and QuAM #
##################
# sourcery skip: avoid-builtin-shadow
experiment = "2D_Rabi_chevron_freq_vs_amplitude"
debug = True
simulate = False
fit_data = True
qubit_list = [0]
digital = []
machine = QuAM("latest_quam.json")
# machine = QuAM(r"C:\Users\T5_2\Desktop\quantum machines demo\data12072022\161848_state_after_2D_Rabi_chevron_freq_vs_duration.json")
# machine = QuAM(r"C:\Users\T5_2\Desktop\quantum machines demo\data12072022\161654_state_after_2D_Rabi_chevron_freq_vs_duration.json")
gate_shape = "drag_cosine"
machine.get_qubit_gate(0, gate_shape).angle2volt.deg180 = 0.4

# machine.qubits[0].wiring.correction_matrix.phase = np.pi/2


machine.qubits[0].f_01 = 5.95e9

machine.drive_lines[0].lo_freq = 6.05e9
use_log_scale = False


# machine.qubits[0].f_01 = 5.98e9

machine.get_qubit_gate(0, gate_shape).length = 200e-9

config = machine.build_config(digital, qubit_list, gate_shape)


# print(config["mixers"]["mixer_drive_line0"][0]["correction"])
# config['mixers']['mixer_drive_line0'][0]['correction'] = [1/np.sqrt(2), 1/np.sqrt(2), -1/np.sqrt(2), 1/np.sqrt(2)]
config["mixers"]["mixer_drive_line0"][0]["correction"] = [1, 0, 0, 1]

###################
# The QUA program #
###################
n_avg = 4e2

# Frequency scan
freq_span = 200e6
df = 2e6
freq = [
    np.arange(
        machine.get_qubit_IF(i) - freq_span,
        machine.get_qubit_IF(i) + freq_span + df / 2,
        df,
    )
    for i in qubit_list
]
a_min = 0.002
a_min = 0
a_max = 1
da = 0.02
if use_log_scale:
    amplitudes = np.logspace(
        np.log10(a_min), np.log10(a_max), int((a_max - a_min) / da)
    )
else:
    amplitudes = np.arange(a_min, a_max + da / 2, da)


# QUA program
with program() as rabi:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    f = declare(int)
    a = declare(fixed)

    for i, q in enumerate(qubit_list):
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        set_dc_offset(
            f"{machine.qubits[q].name}_flux",
            "single",
            machine.get_flux_bias_point(q, "readout").value,
        )

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(a, amplitudes)):
                with for_(*from_array(f, freq[i])):
                    update_frequency(machine.qubits[q].name, f)
                    play("x180" * amp(a), machine.qubits[q].name)
                    align()
                    measure(
                        "readout",
                        machine.readout_resonators[q].name,
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", I[i]),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q[i]),
                    )
                    wait_cooldown_time(5 * machine.qubits[q].t1, simulate)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i, q in enumerate(qubit_list):
            I_st[i].buffer(len(freq[i])).buffer(len(amplitudes)).average().save(f"I{q}")
            Q_st[i].buffer(len(freq[i])).buffer(len(amplitudes)).average().save(f"Q{q}")
            n_st[i].save(f"iteration{q}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=20000)
    job = qmm.simulate(config, rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    x_data, y_data = [], []
    readout_data = []

    qm = qmm.open_qm(config)
    job = qm.execute(rabi)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    figures = []
    for i, q in enumerate(qubit_list):
        id = uuid.uuid1()

        print(f"Qubit {str(q)}")
        qubit_data[i]["iteration"] = 0

        # Live plotting
        if debug:
            fig = plt.figure(figsize=(14, 6))
            interrupt_on_close(fig, job)
            figures.append(fig)

        # Get results from QUA program
        my_results = fetching_tool(
            job, [f"I{q}", f"Q{q}", f"iteration{q}"], mode="live"
        )
        while my_results.is_processing() and qubit_data[i]["iteration"] < n_avg - 1:

            # Fetch results
            data = my_results.fetch_all()
            qubit_data[i]["I"] = data[0]
            qubit_data[i]["Q"] = data[1]
            qubit_data[i]["iteration"] = data[2]

            # Progress bar
            progress_counter(
                qubit_data[i]["iteration"], n_avg, start_time=my_results.start_time
            )

            ################################### save data ###################################
            x_data_i = Valueclass(
                name="Microwave drive frequency",
                unit="Hz",
                value=freq[i] + machine.drive_lines[q].lo_freq,
            )

            y_data_i = Valueclass(
                name="Microwave drive amplitude",
                unit="V",
                value=amplitudes
                * machine.get_qubit_gate(q, gate_shape).angle2volt.deg180,
            )

            readout_data_i = Valueclass(
                name="Readout signal",
                unit="V",
                value=qubit_data[i]["I"] + 1j * qubit_data[i]["Q"],
            )

            ################################### metadata ###################################

            # get start time, now and calculate duration
            time_start = datetime.fromtimestamp(my_results.start_time)
            time_now = datetime.now()
            duration = time_now - time_start

            # Create metadata dict
            genereal_dict = {
                "Experiment name": experiment,
                "Qubit #": q,
                "Total number of averages": n_avg,
                "Current number of averages": qubit_data[i]["iteration"],
            }

            metadata_dict = {
                "Staring time": time_start.strftime("%Y-%m-%d %H:%M:%S"),
                "Duration": duration,
                "Id": id,
            }

            metadata = {
                "General": genereal_dict,
                "Qubit": {},
                "Fitting report": {},
                "Metadata": metadata_dict,
            }

            # Convert metadata dict to pretty string
            def dict2str(d, indent=0, linespace=True):
                res = ""
                for key, value in d.items():
                    if isinstance(value, dict):
                        if not value:
                            continue

                        res += "  " * indent + str(key) + ":\n"
                        res += dict2str(value, indent + 1, linespace=False)
                    else:
                        res += "  " * indent + str(key) + ": " + str(value) + "\n"
                    if linespace:
                        res += "\n"
                return res

            ################################### live plot ###################################
            fig = live_plot(
                x=x_data_i, y=y_data_i, z=readout_data_i, metadata=metadata, fig=fig
            )

        x_data.append(x_data_i)
        y_data.append(y_data_i)
        readout_data.append(readout_data_i)

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
