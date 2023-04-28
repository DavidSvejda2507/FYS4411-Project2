#include <cassert>
#include <iostream>
#include <memory>

#include "../WaveFunctions/wavefunction.h"
#include "../particle.h"
#include "harmonicoscillator.h"

using std::cout;
using std::endl;

// No interaction
// Spherical geometry

HarmonicOscillator::HarmonicOscillator(double omega)
{
    assert(omega > 0);
    m_omega = omega;
}

double HarmonicOscillator::computeLocalEnergy(
    class WaveFunction &waveFunction,
    std::vector<std::unique_ptr<class Particle>> &particles)
{
    double r2 = 0;
    unsigned int N = particles.size();
    for (unsigned int i = 0; i < N; i++)
    {
        auto position = particles[i]->getPosition();
        for (unsigned int j = 0; j < particles[0]->getNumberOfDimensions(); j++)
            r2 += position[j] * position[j];
    }
    // m = omega = 1
    double potentialEnergy = 0.5 * r2;
    double kineticEnergy = waveFunction.computeDoubleDerivative(particles) * -0.5;
    return (kineticEnergy + potentialEnergy) / N;
}
