import dataanalyzer as da
from dataanalyzer import Plotter, Fitter, load_labber_file, fitmodels, Valueclass
import numpy as np
import matplotlib.pyplot as plt

# Load data
path = r"A:\Labber\20221114_Soprano_V2_CharacterizationCooldown\2022\11\Data_1117\q2_T2_inital_scan_3.hdf5"
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


test = Valueclass(name="test", value=[1, 2, 23, 4, 234], unit="V")

freq = np.stack([freq, freq], axis=1)

empty_valueclass = Valueclass(name="empty", value=[], unit="V")
value_i = []

for _ in range(5):
    random = np.random.rand(10).tolist()
    value_i += [random]
    empty_valueclass.value = value_i

empty_valueclass
