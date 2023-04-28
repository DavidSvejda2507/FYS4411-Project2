#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Math/random.h"
#include "WaveFunctions/wavefunction.h"
#include "metropolis.h"
#include "particle.h"

#include <iostream>
using std::cout;
using std::endl;

Metropolis::Metropolis(std::unique_ptr<class Random> rng)
    : MonteCarlo(std::move(rng))
{
}

bool Metropolis::step(
    double stepLength,
    class WaveFunction &waveFunction,
    std::vector<std::unique_ptr<class Particle>> &particles)
{
    /* Perform the actual Metropolis step: Choose a particle at random and
     * change its position by a random amount, and check if the step is
     * accepted by the Metropolis test (compare the wave function evaluated at
     * this new position with the one at the old position).
     */
    int index = m_rng->nextInt(0, particles.size() - 1);

    auto step = std::vector<double>();
    for (unsigned int i = 0; i < particles[0]->getNumberOfDimensions(); i++)
    {
        step.push_back(m_rng->nextGaussian(0, stepLength));
    }
    double q = waveFunction.phiRatio(particles, index, step);

// #define PhiRatioDEBUG
#ifdef PhiRatioDEBUG
    double phiOld = waveFunction.evaluate(particles);
    waveFunction.adjustPosition(particles, index, step);
    particles[index]->adjustPosition(step);
    double phiNew = waveFunction.evaluate(particles);
    auto backStep = std::vector<double>();
    for (unsigned int i = 0; i < step.size(); i++)
        backStep.push_back(-step[i]);
    waveFunction.adjustPosition(particles, index, backStep);
    particles[index]->adjustPosition(backStep);
    std::cout << "phi old: " << phiOld << "\tphi new: " << phiNew << "\tError: " << (phiNew * phiNew / (phiOld * phiOld) - q) / q << "\tq: " << q << std::endl;
#endif

    if (m_rng->nextDouble() < q)
    {
        waveFunction.adjustPosition(particles, index, step);
        particles[index]->adjustPosition(step);

        return true;
    }

    return false;
}
