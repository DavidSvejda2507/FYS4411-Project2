#pragma once

#include <memory>

#include "wavefunction.h"

class TestWavefunction : public WaveFunction
{
public:
    TestWavefunction(std::unique_ptr<WaveFunction> wavefunction);

    void InitialisePositions(std::vector<std::unique_ptr<class Particle>> &particles);
    void adjustPosition(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> step);
    double evaluate(std::vector<std::unique_ptr<class Particle>> &particles);
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles);
    std::vector<double> quantumForce(std::vector<std::unique_ptr<class Particle>> &particles, int index);
    std::vector<double> quantumForceMoved(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step);
    double phiRatio(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step);
    std::vector<double> getdPhi_dParams(std::vector<std::unique_ptr<class Particle>> &particles);

    virtual int getNumberOfParameters() { return m_wavefunc->getNumberOfParameters(); }
    virtual const std::vector<double> &getParameters() { return m_wavefunc->getParameters(); }

private:
    std::unique_ptr<class WaveFunction> m_wavefunc;
};
