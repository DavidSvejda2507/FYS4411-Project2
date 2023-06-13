#include <cassert>
#include <iostream>
#include <math.h>
#include <memory>

#include "../WaveFunctions/wavefunction.h"
#include "../particle.h"
#include "harmonicoscillatorcoulomb.h"

#include "toggles.h"

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
#ifdef CoulombOptimisation
    double r2 = 0;
    int N = particles.size();
    for (int i = 0; i < N; i++)
    {
        r2 += m_distances[0][i][i];
    }
    double interactionEnergy = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
        {
            interactionEnergy += m_distances[0][i][j];
        }
    }
#else
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
#endif
    // double r2_ = 0;
    // std::vector<double> position, position2;
    // for (int i = 0; i < N; i++)
    // {
    //     position = particles[i]->getPosition();
    //     for (unsigned int i = 0; i < particles[0]->getNumberOfDimensions(); i++)
    //         r2_ += position[i] * position[i];
    // }

    // std::cout << "Potential quick: " << r2 << "\tslow: " << r2_ << "\tdiff: " << r2 - r2_ << std::endl;
    // // m = omega = 1
    // double interactionEnergy_ = 0;
    // for (int i = 0; i < N; i++)
    // {
    //     position = particles[i]->getPosition();
    //     for (int j = 0; j < i; j++)
    //     {
    //         position2 = particles[j]->getPosition();
    //         r2_ = 0;
    //         for (unsigned int k = 0; k < particles[0]->getNumberOfDimensions(); k++)
    //             r2_ += (position[k] - position2[k]) * (position[k] - position2[k]);
    //         interactionEnergy_ += 1 / std::sqrt(r2_);
    //     }
    // }
    // std::cout << "Interaction quick: " << interactionEnergy << "\tslow: " << interactionEnergy_ << "\tdiff: " << interactionEnergy - interactionEnergy_ << std::endl;

    double potentialEnergy = 0.5 * r2 * m_omega;
    double kineticEnergy = waveFunction.computeDoubleDerivative(particles) * -0.5;

    return (kineticEnergy + potentialEnergy + interactionEnergy) / N;
}
