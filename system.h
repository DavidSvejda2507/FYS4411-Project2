#pragma once

#include <memory>
#include <vector>

class System
{
public:
    System(
        std::unique_ptr<class Hamiltonian> hamiltonian,
        std::unique_ptr<class WaveFunction> waveFunction,
        std::unique_ptr<class MonteCarlo> solver,
        std::vector<std::unique_ptr<class Particle>> particles,
        bool calculateGradient);

    std::unique_ptr<class Sampler> runEquilibrationSteps(
        double stepLength,
        unsigned int numberOfEquilibrationSteps);

    std::unique_ptr<class Sampler> runMetropolisSteps(
        std::unique_ptr<class Sampler> sampler,
        double stepLength,
        unsigned int numberOfMetropolisSteps);

    double computeLocalEnergy();
    const std::vector<double> &getWaveFunctionParameters();
    std::vector<double> getdPhi_dParams();
    std::vector<double> getParticlePosition(int index);

private:
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;
    bool m_calculateGradient;

    std::unique_ptr<class Hamiltonian> m_hamiltonian;
    std::unique_ptr<class WaveFunction> m_waveFunction;
    std::unique_ptr<class MonteCarlo> m_solver;
    std::vector<std::unique_ptr<class Particle>> m_particles;
};
