import numpy as np
import matplotlib.pyplot as plt
import os
from path import OUTPUT_DIR


def get_histogram(N, NX=100, NY=100, NZ=100):
    """get normalized histogram for experiment with N particles.
    maximum element of output is 1"""
    HIST_TOT = np.zeros((NX, NY, NZ), dtype=np.uint32)
    threadnums = np.arange(8)
    for thread in threadnums:
        fname = os.path.join(
            OUTPUT_DIR, "histogram" + str(N) + "_" + str(thread) + ".bin"
        )
        hist = np.fromfile(fname, dtype=np.uint32).reshape((NX, NY, NZ))
        HIST_TOT += hist

    # HIST_TOT=HIST_TOT/np.max(HIST_TOT)
    return HIST_TOT


####PLOT TWO HISTOGRAMS
def plot_hists():
    ticks = np.linspace(0, 100, 6)
    labels = np.linspace(-1.5, 1.5, 6)

    histogram = get_histogram(N=2)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    hist_xy = np.sum(histogram, axis=2)
    hist_xz = np.sum(histogram, axis=1)
    img = []
    img.append(ax[0].imshow(np.transpose(hist_xy)))
    ax[0].set_ylabel("y")
    img.append(ax[1].imshow(np.transpose(hist_xz)))
    ax[1].set_ylabel("z")

    for ax_cur, imgcur in zip(ax, img):
        ax_cur.set_xticks(ticks)
        ax_cur.set_yticks(ticks)
        ax_cur.set_xticklabels(["{:.1f}".format(label) for label in labels])
        ax_cur.set_yticklabels(["{:.1f}".format(label) for label in labels[::-1]])
        ax_cur.set_xlabel("x")
        # fig.colorbar(imgcur, ax=ax_cur, format='%.0e')
    plt.savefig("onebody_xy_xz_100.pdf", bbox_inches="tight")
    plt.show()


def gaussian_fit(n, alphaopt, betaopt):
    # get histograms along xy plane and z direction
    HIST = get_histogram(n)
    hist_xy = np.sum(HIST, axis=2)
    hist_z = np.sum(HIST, axis=(0, 1))
    # flatten to perform linear fit
    hist_xy_flat = hist_xy.reshape(-1)
    # avoid log(0) and log of small numbers
    nnz_idx = np.nonzero(hist_xy_flat > np.exp(7))
    nnz_idx_z = np.nonzero(hist_z > np.exp(5))
    log_hist = np.log(hist_xy_flat[nnz_idx])
    log_hist_z = np.log(hist_z[nnz_idx_z])
    # need the distance from each slot to the center
    size = 100
    min, max = -1.5, 1.5
    # Create a 2D array with x and y coordinates of each slot
    x_coords = np.linspace(min, max, size)
    y_coords = np.linspace(min, max, size)
    zz = np.linspace(min, max, size)
    xx, yy = np.meshgrid(x_coords, y_coords)
    # Compute the distance of each slot from the origin 0,0
    r2 = (xx) ** 2 + (yy) ** 2
    r2 = r2.reshape(-1)
    r2 = r2[nnz_idx]
    z2 = zz[nnz_idx_z] ** 2

    # plt.plot(z2, log_hist_z, linestyle='none', marker='.')
    # fit the gaussian
    coeffs, var = np.polyfit(r2, log_hist, deg=1, cov=True)
    alpha = -coeffs[0] / 2
    coeff_z, varz = np.polyfit(z2, log_hist_z, deg=1, cov=True)
    beta = -coeff_z[0] / (2 * alpha)
    varbeta = np.sqrt(varz[0, 0]) / (2 * alpha)
    print(
        f"${n}$ & $\{alpha:.3f}$ &  \
          ${alphaopt:.5f}$ & \
          ${beta:.4f} $  &\
          ${betaopt:.4f}$  & \
          ${alpha*beta:.4f}$ & \
          ${alphaopt*betaopt:.4f}$  \\\\ "
    )
    # plt.show()
    # get histogram along z axis:


# hist_z = np.sum(HIST_TOT, axis=(0,1))

if __name__ == "__main__":
    print(
        "$N$ & $\\alpha^\\star$ & $\\alpha^{\\text{opt}}$ & \
           $\\beta^\\star$ & $\\beta^{\\text{opt}}$ & \
           $\\alpha^\\star\\beta^\\star$ & $\\alpha^{\\text{opt}}\\beta^{\\text{opt}}$\
          \\\\"
    )
    betaopt = np.array([2.8614, 2.9829, 2.9920])
    alphaopt = np.array([0.4941, 0.4742, 0.4673])
    # for i, n in enumerate([2]):
    #     gaussian_fit(n, alphaopt[i], betaopt[i])
    plot_hists()
