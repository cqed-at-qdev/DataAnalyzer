import dataanalyzer as da
from dataanalyzer import Plotter, Fitter, load_labber_file, fitmodels
import numpy as np
import matplotlib.pyplot as plt

# Load data
path = r"/Users/malthenielsen/Desktop/QDev/Sample data for plotter/q0_punchout_VNA.hdf5"
freq, power, pulse = load_labber_file(path, insepct=True)

freq.unit = ""

# Take real part of data
pulse = pulse.real / 1e-9
pulse.unit = "V"


# Create plotter
plot = Plotter(subplots=(1, 1))
plot.heatmap(freq, power, pulse)
plot.heatmap(freq, power, pulse, keep_colorbar=True)
plot.show()

