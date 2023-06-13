#pragma once

#include <memory>

#include "wavefunction.h"

#include "toggles.h"

class InteractingGaussianFermion : public WaveFunction
{
public:
    // grad_optimisation leads to better performance when calculating the gradient, but worse when not calculating the gradient
    InteractingGaussianFermion(double alpha, double beta = 0.5, double omega = 1, bool grad_optimisation = false);
    ~InteractingGaussianFermion();
    void InitialisePositions(std::vector<std::unique_ptr<class Particle>> &particles);
    void adjustPosition(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> step);
    double evaluate(std::vector<std::unique_ptr<class Particle>> &particles);
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles);
    std::vector<double> quantumForce(std::vector<std::unique_ptr<class Particle>> &particles, int index);
    std::vector<double> quantumForceMoved(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step);
    double phiRatio(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step);
    std::vector<double> getdPhi_dParams(std::vector<std::unique_ptr<class Particle>> &particles);

#ifdef CoulombOptimisation
    std::vector<std::vector<double>> *getDistances();
#endif

private:
    int m_n, m_n_2;
    const int m_dim = 2;
    double m_sqrtAO;
    double **m_invMatrixUp, **m_invMatrixDown;
    std::vector<std::array<int, 2>> m_nxny;

    std::vector<std::vector<std::array<double, 2>>> m_phi_prime;
    std::vector<std::vector<double>> m_phi_primePrime;
    std::vector<std::vector<double>> m_phi_alpha;
    bool m_grad_optimisation;

    std::vector<std::vector<double>> m_distances;
    std::vector<std::vector<double>> m_jPrime;
    std::vector<std::vector<double>> m_jDoublePrime;
    std::vector<std::vector<double>> m_jBeta;
    std::vector<double> m_interForcesJastrow;

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
    double evalPhiAlpha(int i, std::vector<double> const &pos, double phi0, double r2);
    double hermiteAlpha(int n, double pos);
    void arrayValsAlpha(std::vector<double> const &pos, std::vector<double> &output);
    double dotProduct(std::vector<double> &vals, int index);
    std::vector<double> vectorDotProduct(std::vector<std::array<double, 2>> &vals, int index);
    std::vector<double> quantumForceSlater(int index);
    std::vector<double> quantumForceJastrow(std::vector<std::unique_ptr<class Particle>> &particles, int index);
    void updateInverseMatrix(int index, std::vector<double> arrayVals);
    void testInverse(std::vector<std::unique_ptr<class Particle>> &particles);
    void changeAlpha(std::vector<std::unique_ptr<class Particle>> &particles);
    double jPrime(double a, double r);
    double jDoublePrime(double a, double r);
    double jBeta(double a, double r);
};
