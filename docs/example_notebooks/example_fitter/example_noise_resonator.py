from dataanalyzer import Plotter, Fitter, fitmodels, Valueclass
import numpy as np

# Set random seed
r = np.random
r.seed(1)

# Create dommy resonator data
n_points = 1000
noise_factor = 0.0

x_start = 0
x_stop = 1
x = np.linspace(x_start, x_stop, n_points)

n_resonators = 2
n_negative_resonators = 1


# Make polynomial function
poly_order = 1
number = -0.00
pol_params = {
    "c0": r.uniform(number, number),
    "c1": r.uniform(number, number),
    "c2": r.uniform(number, number),
    "c3": r.uniform(number, number),
    "c4": r.uniform(number, number),
}

pol_func = fitmodels.PolynomialModel(degree=poly_order)


# Create background polynomial with noise
y_noise = r.uniform(0, noise_factor, n_points)
y_pol = pol_func.func(x, **dict(list(pol_params.items())[: poly_order + 1]))


# Create resonator data
resonator_params = [
    {
        "amplitude": abs(r.uniform(0.3, 1)),
        "center": abs(r.uniform(x_start, x_stop)),
        "sigma": 0.01 + r.uniform(0.001, 0.01),
    }
    for _ in range(n_resonators)
]

y_res = np.zeros(n_points)
for resonator_param in resonator_params:
    y_res += fitmodels.LorentzianModel().func(x, **resonator_param)

y_data = Valueclass(value=y_pol + y_res + y_noise, name="Simulated data", unit="a.u.")

# Fit data
func = fitmodels.LorentzianModel() + fitmodels.LorentzianModel()

func.models[1]._param_root_names

fit = Fitter(x=x, y=y_data.value, func=func)
x_fit, y_fit, params, report = fit.do_fit()

plot = Plotter()
plot.scatter(x=x, y=y_data.value, label="Data")
plot.plot(x=x_fit, y=y_fit, c="r", label="Fit")
plot.show()
