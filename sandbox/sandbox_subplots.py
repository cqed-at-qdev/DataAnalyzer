import dataanalyzer as da
from dataanalyzer import Plotter, Fitter, load_labber_file, fitmodels
import numpy as np
import matplotlib.pyplot as plt

# Load data
path = r"A:\Labber\20221114_Soprano_V2_CharacterizationCooldown\2022\11\Data_1117\q2_T2_inital_scan_3.hdf5"
path = "/Users/malthenielsen/Desktop/QDev/Sample data for plotter/q2_T2_inital_scan_3.hdf5"
(duration, repetition), (pulse, _) = load_labber_file(path, insepct=False)

# Take real part of data
pulse = pulse.real

# Create plotter
plot = Plotter(subplots=(2, 2))

type(plot.axs)
# Plot multiple traces with repetitions as labels (first 6 repetitions) [first subplot]
for i, rep in enumerate(repetition[:6].value):
    plot.plot(duration, pulse[i], label=f"Rep {rep:.0f}", title="First 6 repetitions")

# Plot average of all traces [second subplot]
plot.plot(duration, pulse.mean(axis=0), label="Average", title="Average of all", ax=(0, 1))

# Plot average of all traces with errorbars [third subplot]
plot.errorbar(
    duration,
    pulse.mean(axis=0),
    yerr=pulse.std(axis=0),
    label="Average",
    title="Average of all with errorbars",
    ax=(1, 0),
)

# Plot average of all traces and fit [fourth subplot]
# func = fitmodels.DampedOscillationModel()
# fit = Fitter(func, duration, pulse[0])
# duration_fit, pulse_fit, fit_params, report = fit.do_fit()

plot.plot(duration, pulse[0], label="Average", title="Average of all with fit", ax=(1, 1))
# plot.plot(
#     duration_fit, pulse_fit, label="Fit", title="Average of all with fit", ax=(1, 1)
# )


plot.show()
