#pragma once
#include <memory>
#include <vector>

#include "toggles.h"

class Hamiltonian
{
public:
    virtual ~Hamiltonian() = default;
    virtual double computeLocalEnergy(
        class WaveFunction &waveFunction,
        std::vector<std::unique_ptr<class Particle>> &particles) = 0;
#ifdef CoulombOptimisation
    void setDistances(std::vector<std::vector<double>> *distances)
    {
        m_distances = distances;
    }

protected:
    std::vector<std::vector<double>> *m_distances;
#endif
};
