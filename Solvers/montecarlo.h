#pragma once

#include <memory>
#include <vector>

class MonteCarlo
{
public:
    MonteCarlo(std::unique_ptr<class Random> rng);
    virtual ~MonteCarlo() = default;

    virtual bool step( // Only placeholder, will always fail
        double stepLength,
        class WaveFunction &waveFunction,
        std::vector<std::unique_ptr<class Particle>> &particles) = 0;

protected:
    std::unique_ptr<class Random> m_rng;
};
