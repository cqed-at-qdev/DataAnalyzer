import numpy as np

import dataanalyzer as da
import numpy as np

if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # import matplotlib.gridspec as gridspec
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # # Use 'presentation.mplstyle' as style file
    # plt.style.use(r"C:\Users\T5_2\Documents\GitHub\DataAnalyzer\dataanalyzer\plotter\plot_styles\presentation.mplstyle")

    freq = np.linspace(6.60, 6.62, 20) * 1e9
    power = np.linspace(-65, -350000000000000000, 20) * 1e-3
    S21 = np.random.rand(20, 20) * 1e-3

    # plt.close("all")
    # plt.clf()

    # # define plot parameters
    # plt.clf()
    # """ Here, we need to set the actual size of the figure we're going for. In our format 8.6cm is single column, double column is 17.2cm"""
    # """ when you create the illustrator pdf make the artwork 8.6cm x whatever you need in y"""

    # # plt.rcParams['figure.figsize'] =(30.0/2.54,15/2.54)
    # # plt.rcParams["figure.figsize"] = (10 / 2.54, 10 / 2.54)  # here I set something reasonable
    # # plt.rc("font", size=10, family="Arial Bold")  # changed this to Helvetica as in all other figures
    # # plt.rc("axes", unicode_minus=False)
    # # plt.rc("pdf", fonttype=42)  # 3 is default but makes paths

    # fig = plt.figure(constrained_layout=False)

    # gs00 = gridspec.GridSpec(1, 1, figure=fig, wspace=1)
    # # you can define ax1 = fig.add_subplot(gs0[0])
    # # ax2 = fig.add_subplot(gs0[1]), this is what I do here
    # ax2 = fig.add_subplot(gs00[0])

    # axins2 = inset_axes(
    #     ax2, width="30%", height="3%", loc="upper right", bbox_to_anchor=(0.0, 0.1, 1, 1), bbox_transform=ax2.transAxes
    # )

    # pcm = ax2.pcolormesh(freq, power, S21, cmap="inferno", rasterized=True)
    # fig.colorbar(pcm, cax=axins2, orientation="horizontal")  # , ticks = [-1, 2, 4])

    # ax2.set_ylabel(r"$P_{VNA}\, {\mathrm{(dBm)}}$", size=9)
    # ax2.set_xlabel(r"$f_{\mathrm{RF}}\,{\mathrm{(GHz)}}$", size=9, color="black")
    # ax2.set_yscale("linear")
    # # ax2.ticklabel_format(axis="both", style="sci")

    # # Set the axis of the colorbar to be on the outside
    # axins2.xaxis.set_ticks_position("top")

    # # Set ylabel of colorbar and rotate 90 degrees
    # axins2.set_ylabel(r"$|S_{21}|\, {\mathrm{(dB)}}$", size=9, color="black")
    # axins2.yaxis.label.set(rotation="horizontal", ha="right", va="center")

    # # Make labels bold
    # axins2.xaxis.label.set_weight("bold")
    # axins2.yaxis.label.set_weight("bold")

    # plt.show()

    da.Plotter.set_style("presentation")

    freq = da.Valueclass(name="Frequency", unit="Hz", value=freq)
    power = da.Valueclass(name="Power", unit="dBm", value=power)
    S21 = da.Valueclass(name="S21", unit="V", value=S21)

    plot = da.Plotter()
    plot.heatmap(freq, power, S21, cmap="inferno")
    # plot.add_xresiuals(freq, power, ax=1)
    plot.show()
