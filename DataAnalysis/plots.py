import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# plt.rcParams['text.usetex'] = True
CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]
# then define a function which plots only the data as function of alpha for a given Nr of particles and returns an axes object
fsize = 10


def single_plot_E_alpha(data, ax=None, Nparticles=1):
    """
    Plot MonteCarlo energy estimates for various number of particles, with errobars corresponding to standard deviation.
    args:
        data: pd.DataFrame containing data, imported from output.txt.
        Nparticles: plot only results corresponding to this number of particles
        ax: plt.Axes where I want to plot the results

    NB: assume there is only one parameter in the wavefunction (for now), don't know what happens when more parameters are given.
    """

    current_data_npart = data[data["n_particles"] == Nparticles]

    nrs_dims = pd.unique(current_data_npart["n_dimensions"])
    for idx, Ndim in enumerate(nrs_dims):
        # select only data related to current number of particles.
        current_data = current_data_npart[current_data_npart["n_dimensions"] == Ndim]
        alphas = current_data["alpha"]
        energy_pts = current_data["E"]
        # energy_pts = current_data["gradient_alpha"]
        std_pts = np.sqrt(current_data["var"])
        ax.errorbar(
            alphas,
            energy_pts,
            std_pts,
            label=f"$D={Ndim}$",
            linestyle="None",
            marker=".",
            capsize=5,
            color=CB_color_cycle[idx],
        )
        # also plot theoretical trial energy
        alpha_plt = np.linspace(alphas.min(), alphas.max(), 200)
        # energy_plt = Etrial(alpha_plt, Ndim, Nparticles)
        # ax.plot(alpha_plt, energy_plt, color=CB_color_cycle[idx])
    ax.set_xlabel("$\\alpha$", fontsize=fsize)
    ax.set_ylabel("trial energy", fontsize=fsize)
    ax.legend(fontsize=fsize)
    ax.grid()
    # ax.set_yscale('log')
    ax.set_title(f"{Nparticles} particle(s)", fontsize=fsize)
    ax.tick_params(axis="both", which="both", labelsize=fsize)
    ax.ticklabel_format(axis="both", style="sci", scilimits=[0, 2])

    return


def plot_E_alpha_Ndim(data, nrs_particles=[1]):
    """Make three subplots in a single figure, each plot for a different number of dimensions."""
    fig, axs = plt.subplots(2, 2, figsize=(6.876, 7.9))
    for i, Nparts in enumerate(nrs_particles):
        single_plot_E_alpha(data, ax=axs[i // 2, i % 2], Nparticles=Nparts)

    return fig, axs


def Etrial(alpha, D, N, omega=1):
    return N * D * (alpha / 2 + (omega**2 / (8 * alpha)))


if __name__ == "__main__":
    #     # import output.txt from simulation as a Pandas dataframe
    #     data = pd.read_csv("Outputs/output.txt", sep="\t")
    #     dims = [1,2,3]
    #     fig, axs = plot_E_alpha_Ndim(data, dims=dims)
    #     plt.show()

    # data = np.genfromtxt("Outputs/output.txt", float, "#", "\t")
    # plt.plot(data[:,6], data[:,9])
    # plt.savefig("test.png")

    # data = pd.read_csv("Outputs/output_temp.txt", sep="\t")
    data = pd.read_csv("Outputs/output.txt", sep="\t")
    print(data)
    Nparts = [2]
    fig, axs = plot_E_alpha_Ndim(data, nrs_particles=Nparts)
    fig.savefig("Plot_Etrial_alpha.pdf", bbox_inches="tight")
    plt.show()
