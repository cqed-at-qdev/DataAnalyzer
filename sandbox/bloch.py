import dataanalyzer as da
import numpy as np
import qutip as qt

# Make random data
x = da.Valueclass(tag="x", name="X", unit="mV", value=np.linspace(0, 10, 100))
y = da.Valueclass(tag="y", name="Y", unit="mV", value=np.linspace(0, 10, 100))
z = da.Valueclass(tag="color", name="Color", unit="mV", value=np.linspace(0, 10, 3))


th = np.linspace(0, 2 * np.pi, 100)
xp = np.cos(th)
yp = np.sin(th)
zp = np.zeros_like(th)

z = da.Valueclass(tag="color", name="Color", unit="mV", value=np.random.random(len(th)))

points = np.array([xp, yp, zp])

# Create plotter
plot = da.Plotter(subplots=(2, 2), interactive=True)
plot.scatter(x, y, title="Scatter plot", ax=(0, 0))
plot.add_bloch_point([1, 0, 0], ax=2)


plot.add_bloch_point(points, z=z, ax=3)


plot.show()
