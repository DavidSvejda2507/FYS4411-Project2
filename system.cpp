#include <cassert>
#include <iostream>
#include <memory>
// #include <bits/stdc++.h>
#include "Hamiltonians/hamiltonian.h"
#include "InitialStates/initialstate.h"
#include "Samplers/sampler.h"
#include "Samplers/samplerFineTune.h"
#include "Solvers/montecarlo.h"
#include "WaveFunctions/wavefunction.h"
#include "particle.h"
#include "system.h"

#include "toggles.h"

void testEquilibration(std::vector<std::unique_ptr<Particle>> &particles, double alpha);

System::System(
    std::unique_ptr<class Hamiltonian> hamiltonian,
    std::unique_ptr<class WaveFunction> waveFunction,
    std::unique_ptr<class MonteCarlo> solver,
    std::vector<std::unique_ptr<class Particle>> particles,
    bool calculateGradient)
{
    m_numberOfParticles = particles.size();
    ;
    m_numberOfDimensions = particles[0]->getNumberOfDimensions();
    m_hamiltonian = std::move(hamiltonian);
    m_waveFunction = std::move(waveFunction);
    m_solver = std::move(solver);
    m_particles = std::move(particles);
    m_waveFunction->InitialisePositions(m_particles);
#ifdef CoulombOptimisation
    // I'm tryint to pass a pointer from the wavefunction to the hamiltonian,
    // but it seems like whatever I do VS code complains
    // As it is it compiles without issues from gcc
    std::vector<std::vector<double>> *temp = m_waveFunction->getDistances();
    m_hamiltonian->setDistances(temp);
#endif
    m_calculateGradient = calculateGradient;
}

std::unique_ptr<class Sampler> System::runEquilibrationSteps(
    double stepLength,
    unsigned int numberOfEquilibrationSteps)
{
    std::unique_ptr<class Sampler> sampler;
    if (m_calculateGradient)
    {
        sampler = std::make_unique<Sampler>(
            m_numberOfParticles,
            m_numberOfDimensions,
            m_waveFunction->getNumberOfParameters());
    }
    else
    {
        sampler = std::make_unique<SamplerFineTune>(
            m_numberOfParticles,
            m_numberOfDimensions,
            0);
    }

    for (unsigned int i = 0; i < numberOfEquilibrationSteps; i++)
    {
        sampler->equilibrationSample(m_solver->step(stepLength, *m_waveFunction, m_particles));
    }

    // TESTING
    // if (m_waveFunction->getNumberOfParameters() > 2) testEquilibration(m_particles, m_waveFunction->getParameters()[2]);

    return sampler;
}

std::unique_ptr<class Sampler> System::runMetropolisSteps(
    std::unique_ptr<class Sampler> sampler,
    double stepLength,
    unsigned int numberOfMetropolisSteps)
{
    for (unsigned int i = 0; i < numberOfMetropolisSteps; i++)
    {
        /* Call solver method to do a single Monte-Carlo step.
         */
        bool acceptedStep = m_solver->step(stepLength, *m_waveFunction, m_particles);

        /* Here you should sample the energy (and maybe other things) using the
         * sampler instance of the Sampler class.
         */
        sampler->sample(acceptedStep, this);
    }
    sampler->transferWaveFunctionParameters(m_waveFunction->getParameters());
    sampler->computeAverages();
    return sampler;
}

double System::computeLocalEnergy()
{
    // Helper function
    return m_hamiltonian->computeLocalEnergy(*m_waveFunction, m_particles);
}

const std::vector<double> &System::getWaveFunctionParameters()
{
    // Helper function
    return m_waveFunction->getParameters();
}

// Return sampled quantities for computation of gradient
std::vector<double> System::getdPhi_dParams()
{
    // Helper function
    return m_waveFunction->getdPhi_dParams(m_particles);
}

void testEquilibration(std::vector<std::unique_ptr<Particle>> &particles, double alpha)
{
    double r2, diff;
    std::vector<double> distances;
    for (unsigned int k = 0; k < particles.size(); k++)
    {
        auto position = particles[k]->getPosition();
        for (unsigned int i = k + 1; i < particles.size(); i++)
        {
            auto position2 = particles[i]->getPosition();
            r2 = 0;
            for (unsigned int j = 0; j < position.size(); j++)
            {
                diff = position[j] - position2[j];
                r2 += diff * diff;
            }
            distances.push_back(sqrt(r2));
        }
    }

    // std::sort(distances.begin(), distances.end());

    std::cout << "Distances:\n";
    for (unsigned int i = 0; i < distances.size(); i++)
        std::cout << distances[i] / alpha << "\t";
    std::cout << "\n";
}

std::vector<double> System::getParticlePosition(int index)
{
    return m_particles[index]->getPosition();
}