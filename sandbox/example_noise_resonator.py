from dataanalyzer import Plotter, Fitter, fitmodels, Valueclass
import numpy as np
import iminuit.cost as cost

####################################################################################################
#                      Setup random seed and number of points                                      #
####################################################################################################
# Set random seed
r = np.random
r.seed(42)

# Create dommy resonator data
n_points = 200
noise_factor = 0.05

# Create x-axis
x_start = -50
x_stop = 200
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
number = 0.000000
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
y_data = Valueclass(value=y_pol + y_res, name="Simulated data", unit="V")  # .remove_baseline()

x_data = Valueclass(value=x * 1e9, name="Freqency", unit="Hz")


####################################################################################################
#                      Fit data                                                                    #
####################################################################################################
# Fit data
# func = fitmodels.PolynomialModel(poly_order)
# fitmodels.LorentzianMultipleModel(3)

sigma_factor = 2.2
data_mask = np.ones(len(x_data), dtype=bool)
func = fitmodels.PolynomialModel(poly_order)
func = fitmodels.GaussianMultipleModel(1)
func = fitmodels.GaussianModel()


# fit = Fitter(x=x_data[data_mask], y=y_data[data_mask], sy=0.1, func=func)
fit = Fitter(x=x_data[data_mask], y=y_data[data_mask], yerr=0.001, model=func)
fit.do_fit()  # x_fit, y_fit, _, _ =

fit.do_fit()

fit.get_guess_array()[0]
fit.get_guess_array()[1]

fit.get_residuals()[0].shape
fit.get_residuals()[1].shape

plot = Plotter()
plot.plot_fit(fit, plot_data=False, plot_guess=True)
plot.show()
