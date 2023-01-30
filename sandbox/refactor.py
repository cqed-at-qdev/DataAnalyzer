from dataanalyzer import Plotter, Fitter, fitmodels, Valueclass
from dataanalyzer.fitter.fitter_classsetup import Fitparam
import numpy as np

func_gauss = fitmodels.GaussianModel()
func_linear = fitmodels.LinearModel()
func = fitmodels.LorentzianModel()  # + fitmodels.GaussianModel()

x_data = np.linspace(0, 100, 200)
y_data = func_linear.func(x_data, slope=0.00, intercept=0.0)
y_data = func_gauss.func(x=x_data, amplitude=2, center=20, sigma=3)
# y_data += func_gauss.func(x=x_data, amplitude=1, center=70, sigma=5)
# y_data += 0.15*np.random.random(1000)


fit = Fitter(x=x_data, y=y_data, func=func)
x_fit, y_fit, param_fit, report = fit.do_fit()

param_values = {key: value["value"] for key, value in param_fit.items()}
y_left = y_data - func.func(x=x_data, **param_values)
fit2 = Fitter(x=x_data, y=y_left, func=fitmodels.GaussianModel())
x_fit2, y_fit2, param_fit2, report2 = fit2.do_fit()

param_values = {key: value["value"] for key, value in param_fit2.items()}
x_array = np.linspace(-10, 30, 10000)
y_array = fitmodels.GaussianModel().func(x=x_array, **param_values)

plot = Plotter()
plot.scatter(x=x_data, y=y_data, label="data")
plot.plot(x=x_fit, y=y_fit, c="orange", label="fit1")
# plot.scatter(x=x_data, y=y_left, label="data")
# plot.plot(x=x_array, y=y_array, c="green", label="fit2")
plot.add_metadata(report)
plot.show()
