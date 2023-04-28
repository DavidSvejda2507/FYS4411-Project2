#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"

class HarmonicOscillator3D : public Hamiltonian
{
public:
    HarmonicOscillator3D(double omega, double gamma);
    double computeLocalEnergy(
        class WaveFunction &waveFunction,
        std::vector<std::unique_ptr<class Particle>> &particles);

private:
    double m_omega;
    double m_gamma;
};
