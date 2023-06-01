#include "sampler.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"
#include "particle.h"
#include "system.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using std::cout;
using std::endl;

// define a dummy default constructor which does nothing.
// this is called when I construct an instance of SamplerFineTune
Sampler::Sampler()
{
    ;
}

Sampler::Sampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    int numberOfWFParams)
{
    m_numberOfParticles = numberOfParticles;
    m_numberOfDimensions = numberOfDimensions;
    m_cumulativeGradientTerms = std::vector<std::vector<double>>(numberOfWFParams, std::vector<double>(2, 0.0));
    m_gradientTerms = std::vector<std::vector<double>>(numberOfWFParams, std::vector<double>(2, 0.0));
}

Sampler::Sampler(std::vector<std::unique_ptr<class Sampler>> &samplers)
{
    // combine the results from all samplers.
    // note: we assume that  every sampler has sampled for the SAME number of steps.
    // otherwise we would need a weighted average

    int numberOfWFParams = samplers[0]->getGradientTerms().size();
    m_numberOfDimensions = samplers[0]->getNdim();
    m_waveFunctionParameters = samplers[0]->getWFparams();
    m_numberOfParticles = samplers[0]->getNparticles();
    m_gradientTerms = std::vector<std::vector<double>>(numberOfWFParams, std::vector<double>(2, 0.0));
    int Nparams = m_gradientTerms.size();
    int Nsamplers = samplers.size();
    for (auto &sampler : samplers)
    {
        // sum sampled energy
        m_energy += sampler->getEnergy();
        m_energy2 += sampler->getEnergy2();
        // sum accepted nr. of steps
        m_stepNumber += sampler->getNSteps();
        m_numberOfAcceptedSteps += sampler->getNAccSteps();
        m_equilibrationStepNumber += sampler->getNStepsEq();
        m_numberOfAcceptedEquilibrationSteps += sampler->getNAccStepsEq();
        // sum sampled gradients
        std::vector<std::vector<double>> cur_gradient = sampler->getGradientTerms();
        for (int j = 0; j < Nparams; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                m_gradientTerms[j][k] += cur_gradient[j][k];
            }
        }
    }
    // obtain averages
    m_energy /= Nsamplers;
    m_energy2 /= Nsamplers;
    for (int j = 0; j < Nparams; j++)
    {
        for (int k = 0; k < 2; k++)
        {
            m_gradientTerms[j][k] /= Nsamplers;
        }
    }
}

void Sampler::equilibrationSample(bool acceptedStep)
{
    m_equilibrationStepNumber++;
    m_numberOfAcceptedEquilibrationSteps += acceptedStep;
}

void Sampler::sample(bool acceptedStep, System *system)
{
    /*sample all the interesting things
     */
    double localEnergy = system->computeLocalEnergy();
    m_cumulativeEnergy += localEnergy;
    m_cumulativeEnergy2 += localEnergy * localEnergy;

    // sample quantities for gradient computation
    std::vector<double> currentDerivatives = system->getdPhi_dParams();

    // update each of the rowas adding current sampled quantity
    for (unsigned int i = 0; i < currentDerivatives.size(); i++)
    {
        m_cumulativeGradientTerms[i][0] += currentDerivatives[i];
        m_cumulativeGradientTerms[i][1] += currentDerivatives[i] * localEnergy;
    }
    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;
}

void Sampler::transferWaveFunctionParameters(std::vector<double> parameters)
{
    m_waveFunctionParameters = parameters;
}

void Sampler::printOutputToTerminal()
{
    auto pa = m_waveFunctionParameters;
    auto p = pa.size();
    std::vector<double> grad = computeGradientEtrial();

    cout << endl;
    cout << "  -- System info -- " << endl;
    cout << " Number of particles  : " << m_numberOfParticles << endl;
    cout << " Number of dimensions : " << m_numberOfDimensions << endl;
    cout << " Number of equilibration Metropolis steps run : 10^" << std::log10(m_equilibrationStepNumber) << endl;
    cout << " Ratio of accepted equilibration steps: " << ((double)m_numberOfAcceptedEquilibrationSteps) / ((double)m_equilibrationStepNumber) << endl;
    cout << " Number of Metropolis steps run : 10^" << std::log10(m_stepNumber) << endl;
    cout << " Ratio of accepted steps: " << ((double)m_numberOfAcceptedSteps) / ((double)m_stepNumber) << endl;
    cout << endl;
    cout << "  -- Wave function parameters -- " << endl;
    cout << " Number of parameters : " << p << endl;
    for (unsigned int i = 0; i < p; i++)
    {
        cout << " Parameter " << i + 1 << " : " << pa.at(i) << endl;
    }
    cout << endl;
    cout << "  -- Results -- " << endl;
    cout << " Energy : " << std::setprecision(8) << m_energy << endl;
    cout << " Variance in the energy : " << m_energy2 - m_energy * m_energy << endl;
    cout << " Gradient of the variational parameters: ";
    for (unsigned int i = 0; i < grad.size(); i++)
    {
        cout << grad[i] << '\t';
    }
    cout << endl
         << endl;
}

void Sampler::printOutputToTerminalShort()
{
    auto pa = m_waveFunctionParameters;
    auto p = pa.size();
    std::vector<double> grad = computeGradientEtrial();

    cout << endl;
    cout << "  -- System info -- " << endl;
    cout << " Ratio of accepted equilibration steps: " << ((double)m_numberOfAcceptedEquilibrationSteps) / ((double)m_equilibrationStepNumber) << endl;
    cout << " Ratio of accepted steps: " << ((double)m_numberOfAcceptedSteps) / ((double)m_stepNumber) << endl;
    cout << endl;
    cout << " Number of parameters : " << p << endl;
    for (unsigned int i = 0; i < p; i++)
    {
        cout << " Parameter " << i + 1 << " : " << pa.at(i) << endl;
    }
    cout << endl;
    cout << "  -- Results -- " << endl;
    cout << std::setprecision(8);
    cout << " Energy : " << m_energy << endl;
    cout << " Variance in the energy : " << m_energy2 - m_energy * m_energy << endl;
    cout << " Gradient of the variational parameters: ";
    for (unsigned int i = 0; i < grad.size(); i++)
    {
        cout << grad[i] << '\t';
    }
    cout << endl
         << endl;
}

void Sampler::computeAverages()
{
    /* Compute the averages of the sampled quantities.
     */
    m_energy = m_cumulativeEnergy / m_stepNumber;
    m_energy2 = m_cumulativeEnergy2 / m_stepNumber;

    for (size_t i = 0; i < m_cumulativeGradientTerms.size(); i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            m_gradientTerms[i][j] = m_cumulativeGradientTerms[i][j] / m_stepNumber;
        }
    }
}

void Sampler::initiateFile(std::string filename)
{
    std::ofstream file(filename, std::ofstream::app);
    file << "n_particles" << '\t'
         << "n_dimensions" << '\t'
         << "n_steps" << '\t'
         << "n_accepted_steps" << '\t'
         << "E" << '\t'
         << "var" << '\t'
         << "alpha" << '\t'
         << "beta" << '\t'
         << "a" << '\t'
         << "gradient_alpha" << '\t'
         << "gradient_beta" << '\t'
         << "gradient_omega" << endl;
    file.close();
}

void Sampler::writeToFile(std::string filename)
{
    std::ofstream file(filename, std::ofstream::app);
    std::vector<double> grad = computeGradientEtrial();
    file << std::setprecision(10);
    file << m_numberOfParticles << '\t'
         << m_numberOfDimensions << '\t'
         << m_stepNumber << '\t'
         << m_numberOfAcceptedSteps << '\t'
         << m_energy << '\t'
         << m_energy2 - m_energy * m_energy; // variance
    for (unsigned int i = 0; i < m_waveFunctionParameters.size(); i++)
        file << '\t' << m_waveFunctionParameters[i];
    for (unsigned int i = 0; i < grad.size(); i++)
        file << '\t' << grad[i];
    file << endl;
    file.close();
}

// gradient of the trial energy
std::vector<double> Sampler::computeGradientEtrial()
{
    int N = m_gradientTerms.size();
    std::vector<double> gradient = std::vector<double>(N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            gradient[i] = 2 * (m_gradientTerms[i][1] - m_energy * m_gradientTerms[i][0]);
        }
    }
    return gradient;
}

void Sampler::writeHistogram()
{
    std::cout << "Warning. Attempted to write histogram during optimization. Exiting now. \n"
              << std::endl;
    exit(1);
}