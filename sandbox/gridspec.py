import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# import numpy as np

fig = plt.figure(figsize=(8, 8))
nn = 6

# will create gridspec of 6 rows, 6 columns
# 1st row will occupy v small heights
# last column will occupy v small widths

sm = 0.01  # the v small width/height
wh = (1.0 - sm) / (nn - 1.0)  # useful width/height

gs = gridspec.GridSpec(
    nn, nn, width_ratios=[*[wh] * (nn - 1), sm], height_ratios=[sm, *[wh] * (nn - 1)]
)

cols, rows = nn, nn
ax = [[0 for i in range(cols)] for j in range(rows)]

for ea in range(nn):
    for eb in range(nn):
        ax[ea][eb] = fig.add_subplot(gs[ea, eb])
        ax[ea][eb].set_xticklabels([])
        ax[ea][eb].set_yticklabels([])
        if eb >= ea:
            ax[ea][eb].remove()

# plot data on some axes
# note that axes on the first row (index=0) are gone
ax[2][0].plot([2, 5, 3, 7])
ax[4][2].plot([2, 3, 7])

# make legend in upper-right axes (GridSpec's first row, last column)
# first index: 0
# second index: nn-1
rx, cx = 0, nn - 1
ax[rx][cx] = fig.add_subplot(gs[rx, cx])
hdl = ax[rx][cx].scatter([10], [10], color="k", s=5, zorder=2, label="Targets")
ax[rx][cx].set_axis_off()
# ax[rx][cx].set_visible(True)  # already True
ax[rx][cx].set_xticklabels([])
ax[rx][cx].set_yticklabels([])

# plot legend
plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right", borderaxespad=0.0)

fig.subplots_adjust(hspace=0.00, wspace=0.00)

plt.show()
