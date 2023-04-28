#pragma once

#include <memory>

#include "wavefunction.h"

class InteractingGaussianFermion : public WaveFunction
{
public:
    InteractingGaussianFermion(double alpha, double beta = 1, double a = 0.0043);
    void InitialisePositions(std::vector<std::unique_ptr<class Particle>> &particles);
    void adjustPosition(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> step);
    double evaluate(std::vector<std::unique_ptr<class Particle>> &particles);
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles);
    std::vector<double> quantumForce(std::vector<std::unique_ptr<class Particle>> &particles, int index);
    std::vector<double> quantumForceMoved(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step);
    double phiRatio(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step);
    std::vector<double> getdPhi_dParams(std::vector<std::unique_ptr<class Particle>> &particles);

private:
    // std::vector<std::vector<double>> m_distances;
    // std::vector<double> m_interForces;
    double jPrime(double r);
    double jDoublePrime(double a, double r);
};
