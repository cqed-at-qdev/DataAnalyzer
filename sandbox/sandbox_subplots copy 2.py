import dataanalyzer as da
from dataanalyzer import Plotter, Fitter, load_labber_file, fitmodels, Valueclass
import numpy as np
import matplotlib.pyplot as plt

# Load data
path = r"A:\Labber\20221114_Soprano_V2_CharacterizationCooldown\2022\11\Data_1117\q2_T2_inital_scan_3.hdf5"
path = "/Users/malthenielsen/Desktop/QDev/Sample data for plotter/q2_T2_inital_scan_3.hdf5"
[freq, power], [pulse, _] = load_labber_file(path, insepct=True)

# Take real part of data
power = power[:6]
pulse = pulse.real
pulse.unit = "V"


# Create plotter
plot = Plotter(subplots=(1, 1))
plot.heatmap(freq, power, pulse)
plot.heatmap(freq, power, pulse, keep_colorbar=True)
plot.show()
