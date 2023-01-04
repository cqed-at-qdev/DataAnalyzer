from dataanalyzer import Plotter, Fitter, fitmodels, load_labber_file
import os

func = fitmodels.DampedOscillationModel()
func.units("x", "y")
func.units(1, 1)
func.param_names
func.symbols(amplitude="Test")
func.symbols(overwrite=True)
