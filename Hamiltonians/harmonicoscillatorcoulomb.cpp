#include <cassert>
#include <iostream>
#include <math.h>
#include <memory>

#include "../WaveFunctions/wavefunction.h"
#include "../particle.h"
#include "harmonicoscillatorcoulomb.h"

using std::cout;
using std::endl;

// No interaction
// Spherical geometry

HormonicOscillatorCoulomb::HormonicOscillatorCoulomb(double omega)
{
    assert(omega > 0);
    m_omega = omega;
}

double HormonicOscillatorCoulomb::computeLocalEnergy(
    class WaveFunction &waveFunction,
    std::vector<std::unique_ptr<class Particle>> &particles)
{
    double r2 = 0;
    unsigned int N = particles.size();
    std::vector<double> position, position2;
    for (unsigned int i = 0; i < N; i++)
    {
        position = particles[i]->getPosition();
        for (unsigned int i = 0; i < particles[0]->getNumberOfDimensions(); i++)
            r2 += position[i] * position[i];
    }
    // m = omega = 1
    double potentialEnergy = 0.5 * r2;
    double kineticEnergy = waveFunction.computeDoubleDerivative(particles) * -0.5;
    double interactionEnergy = 0;
    for (unsigned int i = 0; i < N; i++)
    {
        position = particles[i]->getPosition();
        for (unsigned int j = 0; j < i; j++)
        {
            position2 = particles[j]->getPosition();
            r2 = 0;
            for (unsigned int k = 0; k < particles[0]->getNumberOfDimensions(); k++)
                r2 += (position[k] - position2[k]) * (position[k] - position2[k]);
            interactionEnergy += 1 / std::sqrt(r2);
        }
    }

    return (kineticEnergy + potentialEnergy + interactionEnergy) / N;
}
