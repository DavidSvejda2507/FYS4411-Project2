# Variational Monte Carlo solver for a Fermionic condensate
![Languages](https://img.shields.io/badge/Languages-C%2B%2B%20%7C%20Python-blue)
![Size](https://img.shields.io/github/repo-size/DavidSvejda2507/FYS4411-Project1)

This repository contains the code for the second project of the course [FYS4411 (spring 2023)](https://github.com/CompPhysics/ComputationalPhysics2).
This code was adapted from the code for the first project, which was forked from this template [repository](https://github.com/mortele/variational-monte-carlo-fys4411.git), whose authors are thankfully acknowledged.

### Disclaimer
This README was written for the first project and has not been properly updated for the second project. It may be out of date.
 
### Requirements
The code is written in C++. To compile it, you need to have the [nlopt library](https://nlopt.readthedocs.io/en/latest/) and [OpenMP](https://www.openmp.org/) installed. The data analysis is written in python.

### Compiling and running the project
Provided Cmake is installed, compile by running the script `compile_project` via
```bash
./compile_project
```
This produces an executable called `vmc` in the top directory. It should be run as:
```bash
./vmc Input/simulation_input.txt
```
where `Input/simulation_input.txt` is the file containing the parameters of the simulation.
There are two ways of running the code. 

1. If the input file contains the line `calculateGradients=1`, the code will compute the gradients and run an optimization algorithm for the parameters of the wave function. By default this is LBFGS, but it can be set to be gradient descent with momentum by providing an extra command-line argument `GD`:
    ```bash
        ./vmc Input/simulation_input.txt GD
    ```
    Progress of the optimization is written to `Outputs/output.txt`.

2. If the input file contains the line `calculateGradients=0`, the code will __NOT__ compute the gradients. This is meant to spare the computational costs when the wave function is already optimized. It will output a lot of files in the `Outputs` folder, which contain the sampled energies and the one-body histograms (in binary format).

### Data Analysis
The python scripts for the data analysis are contained in the `DataAnalysis` folder, they assume that a simulation has been run without calculating gradients.

* `onebodydensity.py` computes and plots the histograms and the corresponding gaussian fit.
* `blocking.py` contains the blocking function provided by [Marius Jonsson](https://github.com/computative/block) and uses it to analyse the simulation data.

## Authors
[David Svejda](https://github.com/DavidSvejda2507)<br>

The first project whose code forms the basis of this project was done in collaboration with
[Gianmarco Puleo](https://github.com/giammy00) <br>

