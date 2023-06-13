#pragma once
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
class Sampler
{
public:
    Sampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        int numberOfWFParams);
    Sampler(std::vector<std::unique_ptr<class Sampler>> &samplers);
    Sampler();
    virtual ~Sampler() = default;

    virtual void sample(bool acceptedStep, class System *system);
    void equilibrationSample(bool acceptedStep);
    void transferWaveFunctionParameters(std::vector<double> parameters);
    void printOutputToTerminal();
    void printOutputToTerminalShort();
    void computeAverages();
    // implement getters to combine results and print them to terminal/file
    double getEnergy() { return m_energy; }
    double getEnergy2() { return m_energy2; }
    std::vector<std::vector<double>> getGradientTerms() { return m_gradientTerms; }
    int getNSteps() { return m_stepNumber; }
    int getNStepsEq() { return m_equilibrationStepNumber; }
    int getNAccSteps() { return m_numberOfAcceptedSteps; }
    int getNAccStepsEq() { return m_numberOfAcceptedEquilibrationSteps; }
    int getNdim() { return m_numberOfDimensions; }
    std::vector<double> getWFparams() { return m_waveFunctionParameters; }
    int getNparticles() { return m_numberOfParticles; }

    std::vector<double> computeGradientEtrial();
    void initiateFile(std::string filename);
    void writeToFile(std::string filename);
    virtual void writeHistogram();

protected:
    unsigned int m_stepNumber = 0;
    unsigned int m_equilibrationStepNumber = 0;
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;
    unsigned int m_numberOfAcceptedSteps = 0;
    unsigned int m_numberOfAcceptedEquilibrationSteps = 0;
    double m_energy = 0;
    double m_cumulativeEnergy = 0;
    double m_energy2 = 0;
    double m_cumulativeEnergy2 = 0;

    bool m_initialised = false;
    double m_localEnergy;
    std::vector<double> m_currentDerivatives;
    // sampled cumulative quantities for computing gradient.
    std::vector<std::vector<double>>
        m_cumulativeGradientTerms;
    // averaged quantities to compute gradient
    std::vector<std::vector<double>> m_gradientTerms;
    // compute the gradient of the trial energy wrt variational parameters
    std::vector<double> m_waveFunctionParameters;
};
