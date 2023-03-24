#pragma once
#include <memory>
#include <string>
#include <iostream>

class Sampler {
public:
    Sampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        int numberOfWFParams
        );
    // Sampler(std::vector<Sampler> samplers);


    void sample(bool acceptedStep, class System* system);
    void equilibrationSample(bool acceptedStep);
    void transferWaveFunctionParameters(std::vector<double> parameters);
    void printOutputToTerminal();
    void printOutputToTerminalShort();
    void computeAverages();
    double getEnergy() { return m_energy; }
    std::vector<double> computeGradientEtrial();
    void initiateFile(std::string filename);
    void writeToFile(std::string filename);

private:
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
    //sampled cumulative quantities for computing gradient.
    std::vector<std::vector<double>> m_cumulativeGradientTerms;
    //averaged quantities to compute gradient
    std::vector<std::vector<double>> m_gradientTerms;
    //compute the gradient of the trial energy wrt variational parameters
    std::vector<double> m_waveFunctionParameters;

};
