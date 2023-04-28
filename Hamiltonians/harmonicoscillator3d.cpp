#include <cassert>
#include <iostream>
#include <memory>

#include "../WaveFunctions/wavefunction.h"
#include "../particle.h"
#include "harmonicoscillator3d.h"

using std::cout;
using std::endl;

// No interaction
// Spherical geometry

HarmonicOscillator3D::HarmonicOscillator3D(double omega, double gamma)
{
    assert(omega > 0);
    m_omega = omega;
    m_gamma = gamma;
}

double HarmonicOscillator3D::computeLocalEnergy(
    class WaveFunction &waveFunction,
    std::vector<std::unique_ptr<class Particle>> &particles)
{
    // Here, you need to compute the kinetic and potential energies.
    unsigned int N = particles.size();
    double r2 = 0;
    for (unsigned int i = 0; i < N; i++)
    {
        auto position = particles[i]->getPosition();
        r2 += position[0] * position[0];
        r2 += position[1] * position[1];
        r2 += position[2] * position[2] * m_gamma * m_gamma;
    }
    // m = omega = 1
    double potentialEnergy = 0.5 * r2;
    double kineticEnergy = waveFunction.computeDoubleDerivative(particles) * -0.5;
    return (kineticEnergy + potentialEnergy) / N;
}
