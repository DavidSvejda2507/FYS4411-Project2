#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"

class HormonicOscillatorCoulomb : public Hamiltonian
{
public:
    HormonicOscillatorCoulomb(double omega);
    double computeLocalEnergy(
        class WaveFunction &waveFunction,
        std::vector<std::unique_ptr<class Particle>> &particles);
};
