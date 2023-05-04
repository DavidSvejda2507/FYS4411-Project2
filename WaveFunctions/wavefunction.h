#pragma once
#include <memory>
#include <vector>

class WaveFunction
{
public:
    virtual ~WaveFunction() = default;

    virtual int getNumberOfParameters() { return m_numberOfParameters; }
    virtual const std::vector<double> &getParameters() { return m_parameters; }

    virtual void InitialisePositions(std::vector<std::unique_ptr<class Particle>> &particles) = 0;
    virtual void adjustPosition(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> step) = 0;
    virtual double evaluate(std::vector<std::unique_ptr<class Particle>> &particles) = 0;
    virtual double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles) = 0;
    virtual std::vector<double> quantumForce(std::vector<std::unique_ptr<class Particle>> &particles, int index) = 0;
    virtual std::vector<double> quantumForceMoved(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step) = 0;
    virtual double phiRatio(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step) = 0;
    virtual std::vector<double> getdPhi_dParams(std::vector<std::unique_ptr<class Particle>> &particles) = 0;

protected:
    int m_numberOfParameters = 0;
    std::vector<double> m_parameters = std::vector<double>();
};