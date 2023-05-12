#include <assert.h>
#include <math.h>
#include <memory>

#include "../particle.h"
#include "../system.h"
#include "interactinggaussian.h"
#include "interactinggaussian3d.h"
#include "interactinggaussianfermion.h"
#include "simplegaussian.h"
#include "simplegaussian3d.h"
#include "testwavefunction.h"
#include "wavefunction.h"

#include <iostream>

// #define Nabla2
#define Nabla2_Ratio
// #define Force
// #define ForceMoved
// #define PhiRatio

TestWavefunction::TestWavefunction(std::unique_ptr<WaveFunction> wavefunc)
{
    m_wavefunc = std::move(wavefunc);
}

void TestWavefunction::InitialisePositions(std::vector<std::unique_ptr<class Particle>> &particles)
{
    m_wavefunc->InitialisePositions(particles);
}

void TestWavefunction::adjustPosition(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> step)
{
    m_wavefunc->adjustPosition(particles, index, step);
}

double TestWavefunction::evaluate(std::vector<std::unique_ptr<class Particle>> &particles)
{
    return m_wavefunc->evaluate(particles);
}

double TestWavefunction::computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles)
{
    double nablaAnal = m_wavefunc->computeDoubleDerivative(particles);

#ifdef Nabla2
    // Numerical calculation
    double nabla2 = 0, phi, phi_plus, phi_minus;
    const double dx = 1e-5, dx2_1 = 1 / (dx * dx);
    int n = particles[0]->getNumberOfDimensions();
    std::vector<double> step = std::vector<double>(n, 0);
    phi = m_wavefunc->evaluate(particles);
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        for (int j = 0; j < n; j++)
        {
            step[j] = dx;
            m_wavefunc->adjustPosition(particles, i, step);
            particles[i]->adjustPosition(dx, j);
            phi_plus = m_wavefunc->evaluate(particles);
            step[j] = -2 * dx;
            m_wavefunc->adjustPosition(particles, i, step);
            particles[i]->adjustPosition(-2 * dx, j);
            phi_minus = m_wavefunc->evaluate(particles);
            step[j] = dx;
            m_wavefunc->adjustPosition(particles, i, step);
            particles[i]->adjustPosition(dx, j);
            step[j] = 0;

            nabla2 += (phi_plus + phi_minus - 2 * phi) * dx2_1;
        }
    }
    nabla2 /= phi;

    std::cout << "Analytical: " << nablaAnal << "   \t Numerical: " << nabla2 << "   \t Rel Diff: " << abs((nabla2 - nablaAnal) / nabla2) << "   \t Abs Diff: " << abs(nabla2 - nablaAnal) << std::endl;
#else
#ifdef Nabla2_Ratio
    // Numerical calculation
    double nabla2 = 0, phi_plus, phi_minus;
    const double dx = 1e-5, dx2_1 = 1 / (dx * dx);
    int n = particles[0]->getNumberOfDimensions();
    std::vector<double> step = std::vector<double>(n, 0);
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        for (int j = 0; j < n; j++)
        {
            step[j] = dx;
            phi_plus = sqrt(m_wavefunc->phiRatio(particles, i, step));
            step[j] = -dx;
            phi_minus = sqrt(m_wavefunc->phiRatio(particles, i, step));
            step[j] = 0;

            nabla2 += (phi_plus + phi_minus - 2) * dx2_1;
        }
    }

    std::cout << "Analytical: " << nablaAnal << "   \t Numerical: " << nabla2 << "   \t Rel Diff: " << abs((nabla2 - nablaAnal) / nabla2) << "   \t Abs Diff: " << abs(nabla2 - nablaAnal) << std::endl;
#endif
#endif

    return nablaAnal;
}

std::vector<double> TestWavefunction::quantumForce(std::vector<std::unique_ptr<class Particle>> &particles, int index)
{
    auto forceAnal = m_wavefunc->quantumForce(particles, index);

#ifdef Force
    // Numerical calculation
    double phi, phi_plus, phi_minus, diff = 0;
    const double dx = 1e-5, dx2 = 2 * dx;
    int n = particles[0]->getNumberOfDimensions();
    std::vector<double> step = std::vector<double>(n, 0);
    std::vector<double> force = std::vector<double>(n, 0);
    phi = m_wavefunc->evaluate(particles);
    std::cout << "Force diffs:\t";
    for (int j = 0; j < n; j++)
    {
        step[j] = dx;
        m_wavefunc->adjustPosition(particles, index, step);
        particles[index]->adjustPosition(dx, j);
        phi_plus = m_wavefunc->evaluate(particles);
        step[j] = -2 * dx;
        m_wavefunc->adjustPosition(particles, index, step);
        particles[index]->adjustPosition(-2 * dx, j);
        phi_minus = m_wavefunc->evaluate(particles);
        step[j] = dx;
        m_wavefunc->adjustPosition(particles, index, step);
        particles[index]->adjustPosition(dx, j);
        step[j] = 0;

        force[j] = (phi_plus - phi_minus) / (dx2 * phi);
        std::cout << forceAnal[j] - force[j] << "\t";
        diff += (forceAnal[j] - force[j]) * (forceAnal[j] - force[j]);
    }
    std::cout << "Abs Diff: " << diff << std::endl;
#endif

    return forceAnal;
}

std::vector<double> TestWavefunction::quantumForceMoved(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step_)
{
    auto forceAnal = m_wavefunc->quantumForceMoved(particles, index, step_);

#ifdef ForceMoved
    // Numerical calculation
    double phi, phi_plus, phi_minus, diff = 0;
    const double dx = 1e-5, dx2 = 2 * dx;
    int n = particles[0]->getNumberOfDimensions();
    std::vector<double> step = std::vector<double>(n, 0);
    std::vector<double> force = std::vector<double>(n, 0);
    m_wavefunc->adjustPosition(particles, index, step_);
    particles[index]->adjustPosition(step_);
    phi = m_wavefunc->evaluate(particles);
    std::cout << "Force diffs:\t";
    for (int j = 0; j < n; j++)
    {
        step[j] = dx;
        m_wavefunc->adjustPosition(particles, index, step);
        particles[index]->adjustPosition(dx, j);
        phi_plus = m_wavefunc->evaluate(particles);
        step[j] = -2 * dx;
        m_wavefunc->adjustPosition(particles, index, step);
        particles[index]->adjustPosition(-2 * dx, j);
        phi_minus = m_wavefunc->evaluate(particles);
        step[j] = dx;
        m_wavefunc->adjustPosition(particles, index, step);
        particles[index]->adjustPosition(dx, j);
        step[j] = 0;

        force[j] = (phi_plus - phi_minus) / (dx2 * phi);
        std::cout << forceAnal[j] - force[j] << "\t";
        diff += (forceAnal[j] - force[j]) * (forceAnal[j] - force[j]);
    }
    for (int j = 0; j < n; j++)
        step[j] = -step_[j];
    m_wavefunc->adjustPosition(particles, index, step);
    particles[index]->adjustPosition(step);
    std::cout << "Abs Diff: " << diff << std::endl;
#endif

    return forceAnal;
}

double TestWavefunction::phiRatio(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step_)
{
    auto RatioAnal = m_wavefunc->phiRatio(particles, index, step_);

#ifdef PhiRatio
    // Numerical calculation
    double phi_old, phi_new;
    int n = particles[0]->getNumberOfDimensions();
    std::vector<double> step = std::vector<double>(n, 0);
    for (int j = 0; j < n; j++)
        step[j] = -step_[j];

    phi_old = m_wavefunc->evaluate(particles);
    phi_old *= phi_old;
    m_wavefunc->adjustPosition(particles, index, step_);
    particles[index]->adjustPosition(step_);
    phi_new = m_wavefunc->evaluate(particles);
    phi_new *= phi_new;
    m_wavefunc->adjustPosition(particles, index, step);
    particles[index]->adjustPosition(step);
    std::cout << "Ratio analytical: " << RatioAnal << "\t Ratio numerical: " << phi_new / phi_old
              << "\t Meta ratio: " << phi_new / (phi_old * RatioAnal) << std::endl;
#endif

    return RatioAnal;
}

std::vector<double> TestWavefunction::getdPhi_dParams(std::vector<std::unique_ptr<class Particle>> &particles)
{
    return m_wavefunc->getdPhi_dParams(particles);
}