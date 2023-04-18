from dataanalyzer import Valueclass, Plotter
import matplotlib.pyplot as plt
import numpy as np

x, dx = np.linspace(-4, 4, 60, retstep=True)
x_fine = np.linspace(x[0], x[-1], 1000)

y = np.polyval([-0.5, 0.7, 5, 2, -3], x)

# add some noise
y += np.random.normal(0, 1.5, len(x))

# add some outliers
y[5] = 2
y[30] = -50
y[51] = 100

y = Valueclass(y, unit="V")
x = Valueclass(x, unit="s")

y.savgol_removed_outliers(x)

# plot = Plotter()

# width = dx/4

# converged = False
# mask = np.ones_like(x, dtype=bool)
# while not converged:
#     filtered = y[mask].savgol(x[mask], peak_width=width, return_baseline=True, x_output=x)

#     base_line_removed = y - filtered
#     new_mask = base_line_removed.remove_outliers(return_mask=True)
#     if np.all(new_mask == mask):
#         converged = True
#     mask = new_mask


# filted_fine = y[mask].savgol(x[mask], peak_width=width, return_baseline=True, x_output=x_fine)


# plot.show()
