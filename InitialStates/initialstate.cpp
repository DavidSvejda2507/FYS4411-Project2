#include <cassert>
#include <iostream>
#include <memory>

#include "Math/random.h"
#include "initialstate.h"
#include "omp.h"

std::vector<std::unique_ptr<Particle>> setupRandomUniformInitialState(
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    Random &rng,
    double lengthScale)
{
    assert(numberOfDimensions > 0 && numberOfParticles > 0);

    auto particles = std::vector<std::unique_ptr<Particle>>();

    for (unsigned int i = 0; i < numberOfParticles; i++)
    {
        std::vector<double> pos = std::vector<double>();

        for (unsigned int j = 0; j < numberOfDimensions; j++)
        {
            pos.push_back(rng.nextDouble(-lengthScale, lengthScale));
        }
        particles.push_back(std::make_unique<Particle>(pos));
    }

    return particles;
}

std::vector<std::unique_ptr<Particle>> setupRandomGaussianInitialState(
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    Random &rng,
    double lengthScale)
{
    assert(numberOfDimensions > 0 && numberOfParticles > 0);

    auto particles = std::vector<std::unique_ptr<Particle>>();

    for (unsigned int i = 0; i < numberOfParticles; i++)
    {
        std::vector<double> pos = std::vector<double>();

        for (unsigned int j = 0; j < numberOfDimensions; j++)
        {
            pos.push_back(rng.nextGaussian(0, lengthScale));
        }
        particles.push_back(std::make_unique<Particle>(pos));
    }

    return particles;
}

std::vector<std::unique_ptr<Particle>> setupNonOverlappingGaussianInitialState(
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    Random &rng,
    double lengthScale,
    double dist)
{
    assert(numberOfDimensions > 0 && numberOfParticles > 0);

    auto particles = std::vector<std::unique_ptr<Particle>>();
    double r2, r2ref = dist * dist;
    bool collision;
    while (particles.size() < numberOfParticles)
    {
        std::vector<double> pos = std::vector<double>();
        for (unsigned int j = 0; j < numberOfDimensions; j++)
        {
            pos.push_back(rng.nextGaussian(0, lengthScale));
        }
        collision = false;

        for (unsigned int k = 0; k < particles.size(); k++)
        {
            auto pos2 = particles[k]->getPosition();
            r2 = 0;
            for (unsigned int j = 0; j < numberOfDimensions; j++)
            {
                r2 += (pos[j] - pos2[j]) * (pos[j] - pos2[j]);
            }
            if (r2 < r2ref)
            {
                collision = true;
                break;
            }
        }
        if (!collision)
            particles.push_back(std::make_unique<Particle>(pos));
    }

    return particles;
}