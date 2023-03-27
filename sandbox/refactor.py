import dataanalyzer as da
from dataanalyzer import Plotter, Fitter, load_labber_file, fitmodels
import numpy as np
import matplotlib.pyplot as plt
import uncertainties

# Load data
path = "/Users/malthenielsen/Desktop/QDev/Sample data for plotter/q2_T2_inital_scan_3.hdf5"
(duration, repetition), (pulse, _) = load_labber_file(path, insepct=False)
pulse = pulse[0].real

# Fit the data
func = fitmodels.DampedOscillationModel()


# Define plot function
def plot_error_estimation(error, title=None, duration=duration, pulse=pulse, cost="chi2"):
    _pulse = pulse.copy()
    _pulse.error = error

    _fit = Fitter(func, duration, _pulse, cost_function=cost)
    _fit.do_fit()

    plot = Plotter(set_mpl_style=True)
    plot.errorbar(duration, _pulse, label="Data")
    plot.plot_fit(fit_obejct=_fit, plot_data=False)

    plot.ax.set_title(title or f"Fit with errors of {error}")
    plot.show()

    return _fit  # , _pulse


# Plot different error estimations

std = pulse.std().value[0]
fit_0 = plot_error_estimation(error=0.000000001, cost="chi2_iminuit")
error_est_0 = fit_0.get_residuals().std().value[0]
print(f"Estimated error: {error_est_0}")

fit_01 = plot_error_estimation(
    error=error_est_0,
    title=f"Fit with estimated errors of {error_est_0}",
    cost="chi2_iminuit",
)

_ = fit_01.minuit.draw_contour("amplitude", "frequency")
_ = fit_0.minuit.draw_contour("amplitude", "frequency")

fit_0.minuit.interactive()
fit_01.minuit.interactive()


# Manual error estimation
fit_2 = plot_error_estimation(error=0.0001, cost="chi2_iminuit")
minuit_errors = fit_2.minuit.errors.to_dict()


# Do the same with lmfit
import lmfit
from quantum_fitter import Model, QFit, oddfun_damped_oscillations, oddfun_damped_oscillations_guess
import quantum_fitter as qf

X = duration.value
y = pulse.value

a, T, w, p, c = oddfun_damped_oscillations_guess(X, y)

# fitting
t2 = qf.QFit(X, y, model=Model(oddfun_damped_oscillations))
t2.set_params("T", T)
t2.set_params("A", a)
t2.set_params("c", c)
t2.set_params("omega", w)
t2.set_params("phi", p)

t2.do_fit()

t2.pretty_print(
    plot_settings={
        "x_label": "Sequence duration (\u03BCs)",
        "y_label": r"$V_{H}$" " (\u03BCV)",
        "plot_title": f", Averaged",
        "fit_color": "C4",
    },
    x=0,
)


t2._qmodel.fit(data=y, params=t2._params, x=X)

t2.result.eval_uncertainty()


# Get parameter errors
lmfit_errors = t2.err_params()


# Rename the parameters in the lmfit dict
lmfit_errors["amplitude"] = lmfit_errors.pop("A")
lmfit_errors["decay"] = lmfit_errors.pop("T")
lmfit_errors["frequency"] = lmfit_errors.pop("omega") / (2 * np.pi)
lmfit_errors["offset"] = lmfit_errors.pop("c")

# Plot the errors
plt.figure()
plt.scatter(minuit_errors.keys(), minuit_errors.values(), label="iminuit")
plt.scatter(lmfit_errors.keys(), lmfit_errors.values(), label="lmfit")
plt.legend()
plt.show()

# pop frequency on both
minuit_errors.pop("frequency")
lmfit_errors.pop("frequency")

# Compare the errors
plt.figure()
plt.scatter(minuit_errors.keys(), minuit_errors.values(), label="iminuit")
plt.scatter(lmfit_errors.keys(), lmfit_errors.values(), label="lmfit")
plt.legend()
plt.show()


# Difference between the errors
diff = {key: minuit_errors[key] - lmfit_errors[key] for key in minuit_errors.keys()}

plt.figure()
plt.scatter(diff.keys(), diff.values(), label="diff")

plt.xlabel("Parameter")
plt.ylabel("Difference in error")

plt.legend()
plt.show()

# Make simpulated data with errors, folowing a normal distribution
amplitude = 2
phi = 0
offset = 0
frequency = 0.1
noise = 0.1
N = 1000

np.random.seed(42)
x = np.linspace(0, 10, N)
y = amplitude * np.sin(2 * np.pi * frequency * x + phi) + offset + noise * np.random.normal(size=N)
yerr = 0.1 * np.ones_like(x)


# Convert to valueclass
xv = da.Valueclass(x, name="x")
yv = da.Valueclass(y, name="y", error=yerr)


# Fit the data (with iminuit)
func = fitmodels.OscillationModel()
fit = Fitter(func, xv, yv, cost_function="chi2_iminuit")
fit_x, fit_y, _, _ = fit.do_fit()

# Plot the fit
plot = Plotter(set_mpl_style=True)
plot.errorbar(x, y, yerr=yerr, label="Data")
plot.plot_fit(fit_obejct=fit, plot_data=False)
plot.ax.set_title("Fit with errors")
plot.show()


# Fit the data (with lmfit)
import lmfit


def f(x, amplitude, frequency, phi, offset):
    return amplitude * np.sin(2 * np.pi * frequency * x + phi) + offset


# fitting
