#pragma once

#include <memory>

#include "wavefunction.h"

class InteractingGaussianFermion : public WaveFunction
{
public:
    InteractingGaussianFermion(double alpha, double beta = 0.5, double omega = 1);
    ~InteractingGaussianFermion();
    void InitialisePositions(std::vector<std::unique_ptr<class Particle>> &particles);
    void adjustPosition(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> step);
    double evaluate(std::vector<std::unique_ptr<class Particle>> &particles);
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles);
    std::vector<double> quantumForce(std::vector<std::unique_ptr<class Particle>> &particles, int index);
    std::vector<double> quantumForceMoved(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step);
    double phiRatio(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step);
    std::vector<double> getdPhi_dParams(std::vector<std::unique_ptr<class Particle>> &particles);

private:
    int m_n, m_n_2;
    double m_sqrtAO;
    double **m_invMatrixUp, **m_invMatrixDown;
    std::vector<std::array<int, 2>> m_nxny;
    // std::vector<std::vector<double>> m_distances;
    // std::vector<double> m_interForces;
    double evalPhi(int i, std::vector<double> const &pos, double phi0);
    double hermite(int n, double pos);
    void arrayVals(std::vector<double> const &pos, std::vector<double> &output);
    std::array<double, 2> evalPhiPrime(int i, std::vector<double> const &pos, double phi0);
    double hermitePrime(int n, double pos);
    void arrayValsPrime(std::vector<double> const &pos, std::vector<std::array<double, 2>> &output);
    double evalPhiPrimePrime(int i, std::vector<double> const &pos, double phi0);
    double hermitePrimePrime(int n, double pos);
    void arrayValsPrimePrime(std::vector<double> const &pos, std::vector<double> &output);
    double dotProduct(std::vector<double> &newVals, int index);
    std::vector<double> vectorDotProduct(std::vector<std::array<double, 2>> &newVals, int index);
    void updateInverseMatrix(int index, std::vector<double> arrayVals);
    void testInverse(std::vector<std::unique_ptr<class Particle>> &particles);
    double jPrime(double r);
    double jDoublePrime(double a, double r);
};
