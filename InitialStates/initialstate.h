#pragma once

#include <memory>
#include <vector>

#include "../particle.h"
#include "Math/random.h"

std::vector<std::unique_ptr<Particle>> setupRandomUniformInitialState(
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    Random &randomEngine,
    double lengthScale);

std::vector<std::unique_ptr<Particle>> setupRandomGaussianInitialState(
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    Random &randomEngine,
    double lengthScale);

std::vector<std::unique_ptr<Particle>> setupNonOverlappingGaussianInitialState(
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    Random &randomEngine,
    double lengthScale,
    double dist = 0.0043);
