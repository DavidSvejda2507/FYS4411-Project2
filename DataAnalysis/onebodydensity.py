import numpy as np
import matplotlib.pyplot as plt
import os
from path import OUTPUT_DIR


def get_histogram(N_particles, N_threads=8, shape=(100, 100)):
    """get normalized histogram for experiment with N particles.
    maximum element of output is 1"""
    output = np.zeros(shape, dtype=np.uint32)
    for thread in range(N_threads):
        fname = os.path.join(
            OUTPUT_DIR, "histogram" + str(N_particles) + "_" + str(thread) + ".bin"
        )
        hist = np.fromfile(fname, dtype=np.uint32).reshape(shape)
        output += hist

    # output=output/np.max(output)
    return output


####PLOT TWO HISTOGRAMS
def plot_hists(histogram, axes, filename):
    ticks = np.linspace(0, 100, 6)
    labels = np.linspace(-1.5, 1.5, 6)
    labels = ["{:.1f}".format(label) for label in labels]

    fig, ax = plt.subplots(1, len(axes), figsize=(5 * len(axes) - 1, 4))
    if len(axes) == 1:
        ax = [ax]
    for dir, axis in zip(axes, ax):
        hist_ax = np.sum(histogram, axis=dir)
        img = axis.imshow(np.transpose(hist_ax))
        axis.set_ylabel("y")

        axis.set_xticks(ticks)
        axis.set_xticklabels(labels)
        axis.set_yticks(ticks)
        axis.set_yticklabels(labels[::-1])
        axis.set_xlabel("x")
        # fig.colorbar(img, ax=axis, format='%.0e')
    fig.savefig(filename, bbox_inches="tight")
    fig.show()


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


if __name__ == "__main__":
    # print(
    #     "$N$ & $\\alpha^\\star$ & $\\alpha^{\\text{opt}}$ & \
    #        $\\beta^\\star$ & $\\beta^{\\text{opt}}$ & \
    #        $\\alpha^\\star\\beta^\\star$ & $\\alpha^{\\text{opt}}\\beta^{\\text{opt}}$\
    #       \\\\"
    # )
    betaopt = np.array([2.8614, 2.9829, 2.9920])
    alphaopt = np.array([0.4941, 0.4742, 0.4673])
    # for i, n in enumerate([2]):
    #     gaussian_fit(n, alphaopt[i], betaopt[i])

    histogram = get_histogram(6, 8, (100, 100))
    # plot_hists(histogram, [(2,), (1,)], "onebody_xy_xz_100.pdf")
    plot_hists(histogram, [()], "onebody_100.pdf")
