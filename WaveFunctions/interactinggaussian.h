#pragma once

#include <memory>

#include "wavefunction.h"

class InteractingGaussian : public WaveFunction
{
public:
    InteractingGaussian(double alpha, double beta = 1, double a = 0.0043);
    void InitialisePositions(std::vector<std::unique_ptr<class Particle>> &particles);
    void adjustPosition(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> step);
    double evaluate(std::vector<std::unique_ptr<class Particle>> &particles);
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles);
    std::vector<double> quantumForce(std::vector<std::unique_ptr<class Particle>> &particles, int index);
    std::vector<double> quantumForceMoved(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step);
    double phiRatio(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step);
    std::vector<double> getdPhi_dParams(std::vector<std::unique_ptr<class Particle>> &particles);
    double uPrime_r(double r);
    double uDoublePrime(double r);

private:
    std::vector<std::vector<double>> m_distances;
    std::vector<double> m_interForces;
    void slowDoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles, double nabla2_);
    void slowQuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &force);
};
