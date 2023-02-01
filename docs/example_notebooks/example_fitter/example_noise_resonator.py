from dataanalyzer import Plotter, Fitter, fitmodels, Valueclass
import numpy as np

####################################################################################################
#                      Setup random seed and number of points                                      #
####################################################################################################
# Set random seed
r = np.random
r.seed(2)

# Create dommy resonator data
n_points = 1000
noise_factor = 0.0

# Create x-axis
x_start = 0
x_stop = 1000
x = np.linspace(x_start, x_stop, n_points)


####################################################################################################
#                      Create resonator data with noise                                            #
####################################################################################################
n_resonators = 1
n_negative_resonators = 1

# Create resonator function
func_res = fitmodels.GaussianMultipleModel(n_resonators)

# Make polynomial function for background
poly_order = 2
number = 0.000001
pol_params = {
    "c0": r.uniform(-number, number),
    "c1": r.uniform(-number, number),
    "c2": r.uniform(-number, number),
    "c3": r.uniform(-number, number),
    "c4": r.uniform(-number, number),
}

pol_func = fitmodels.PolynomialModel(poly_order)

# Create background polynomial with noise
y_noise = r.uniform(0, noise_factor, n_points)
y_pol = pol_func.func(x, **dict(list(pol_params.items())[: poly_order + 1]))


# Create resonator data
resonator_params = {}
for i in range(n_resonators):
    number = f"_{i + 1}" if n_resonators > 1 else ""
    resonator_params[f"amplitude{number}"] = abs(r.uniform(0.2, 0.3))
    resonator_params[f"center{number}"] = abs(r.uniform(x_start, x_stop))
    resonator_params[f"sigma{number}"] = 0.01 + r.uniform(10, 30)


# Create resonator data
y_res = func_res.func(x, **resonator_params)
y_data = Valueclass(value=y_pol + y_res + y_noise, name="Simulated data", unit="V")
x_data = Valueclass(value=x, name="x-axis", unit="s")


####################################################################################################
#                      Fit data                                                                    #
####################################################################################################
# Fit data
func = fitmodels.PolynomialModel(poly_order)
# func =  fitmodels.GaussianMultipleModel(n_resonators)
func.parameters

fit = Fitter(x=x_data, y=y_data, func=func)

fit.func.parameters
plot = Plotter()
plot.plot_fit(fit_obejct=fit, plot_metadata=True, plot_residuals=True)
plot.show()
