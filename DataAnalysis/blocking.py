import numpy as np
import matplotlib.pyplot as plt
import os
from path import OUTPUT_DIR


def block(x):
    # preliminaries
    d = np.log2(len(x))
    if d - np.floor(d) != 0:
        print("Warning: Data size = %g, is not a power of 2." % np.floor(2**d))
        print("Truncating data to %g." % 2 ** np.floor(d))
        x = x[: 2 ** int(np.floor(d))]
    d = int(np.floor(d))
    n = 2**d
    s, gamma = np.zeros(d), np.zeros(d)
    mu = np.mean(x)

    # estimate the auto-covariance and variances
    # for each blocking transformation
    for i in np.arange(0, d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n) ** (-1) * np.sum((x[0 : (n - 1)] - mu) * (x[1:n] - mu))
        # estimate variance of x
        s[i] = np.var(x)
        # perform blocking transformation
        x = 0.5 * (x[0::2] + x[1::2])

    # generate the test observator M_k from the theorem
    M = (np.cumsum(((gamma / s) ** 2 * 2 ** np.arange(1, d + 1)[::-1])[::-1]))[::-1]
    # fmt: off
    # we need a list of magic numbers
    q =np.array([6.634897,  9.210340,  11.344867, 13.276704, 15.086272, 
              16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
              24.724970, 26.216967, 27.688250, 29.141238, 30.577914, 
              31.999927, 33.408664, 34.805306, 36.190869, 37.566235,
              38.932173, 40.289360, 41.638398, 42.979820, 44.314105, 
              45.641683, 46.962942, 48.278236, 49.587884, 50.892181,])
    # fmt: on

    # use magic to determine when we should have stopped blocking
    for k in np.arange(0, d):
        if M[k] < q[k]:
            break
    if k >= d - 1:
        print("Warning: Use more data")
    return s[k] / 2 ** (d - k)


if __name__ == "__main__":
    print(f"$N$ & \t\t $E$ & \t\t $\\sigma$ & $\\eta$ &  var$(E/N)$ \\\\")
    N_part_list = [2, 6, 12, 20]
    threadnums = np.arange(8)
    # import sampled energies
    energies = np.empty(0)
    errors = np.empty(0)
    for N_particles in N_part_list:
        all_energies = np.empty(0)
        for thread in threadnums:
            filename = (
                "sampledEnergies_" + str(N_particles) + "_" + str(thread) + ".bin"
            )
            x = np.fromfile(os.path.join(OUTPUT_DIR, filename), dtype=float)
            all_energies = np.append(all_energies, x)

        avg_energy = np.mean(all_energies)
        variance_energy = np.var(all_energies)
        # block gives variance, want std.
        error = np.sqrt(block(x))
        energies = np.append(energies, avg_energy)
        errors = np.append(errors, error)
        print(
            f"{N_particles} & \t {avg_energy:.6f} & \t {error:.1e} & {avg_energy/(1+np.sqrt(2)):.6f} & \
              {variance_energy:.1e} \\\\"
        )

    # plt.errorbar(N_part_list, energies, yerr=errors, linestyle='none', marker='.', capsize=10 )
    # plt.xlabel("number of particles")
    # plt.ylabel("energy estimate")
    # plt.grid(visible=True)
    # #plt.savefig("energy_estimates.pdf", bbox_inches='tight')
    # plt.show()
