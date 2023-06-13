#include <math.h>
#include <memory>
#include <string>
#include <vector>

#include "Math/random.h"
#include "WaveFunctions/wavefunction.h"
#include "metropolisHastings.h"
#include "particle.h"

#include <iostream>

MetropolisHastings::MetropolisHastings(std::unique_ptr<class Random> rng)
    : MonteCarlo(std::move(rng))
{
}

bool MetropolisHastings::step(
    double dt,
    class WaveFunction &waveFunction,
    std::vector<std::unique_ptr<class Particle>> &particles)
{
    double temp;
    double sqrtdt = sqrt(dt);
    int index = m_rng->nextInt(0, particles.size() - 1);
    auto force = waveFunction.quantumForce(particles, index);
    auto noise = std::vector<double>();
    auto step = std::vector<double>();
    for (unsigned int i = 0; i < force.size(); i++)
    {
        noise.push_back(m_rng->nextGaussian(0, 1) * sqrtdt);
        step.push_back(force[i] * dt * 0.5 + noise[i]);
        if (step[i] > 10. || step[i] < -10.)
        {
            return false;
        }
    }
    auto forceY = waveFunction.quantumForceMoved(particles, index, step);

    // argument of the greensfunction exponent: -(y-x-D*dt*F(x))^2 / 4*D*dt
    // difference: (+(y-x-D*dt*F(x))^2-(x-y-D*dt*F(y))^2) / 4*D*dt
    // difference: (+(step-D*dt*F(x))^2-(-step-D*dt*F(y))^2) / 4*D*dt
    // difference: (+(noise)^2-(-step-D*dt*F(y))^2) / 4*D*dt
    // difference: (+(noise)^2-(step+D*dt*F(y))^2) / 4*D*dt

    double greensDiff = 0;
    for (unsigned int i = 0; i < noise.size(); i++)
        greensDiff += noise[i] * noise[i];
    for (unsigned int i = 0; i < step.size(); i++)
    {
        temp = step[i] + forceY[i] * dt * 0.5;
        greensDiff -= temp * temp;
    }
    double hastingsArticle = waveFunction.phiRatio(particles, index, step) * exp(greensDiff / (2 * dt));

    if (m_rng->nextDouble() < (hastingsArticle))
    {
        waveFunction.adjustPosition(particles, index, step);
        particles[index]->adjustPosition(step);
        return true;
    }
    return false;
}
