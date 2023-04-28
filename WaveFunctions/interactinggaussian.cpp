#include <memory>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <string>

#include "interactinggaussian.h"
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"

#include <iostream>

InteractingGaussian::InteractingGaussian(double alpha, double beta, double a)
{
    assert(alpha > 0); // If alpha == 0 then the wavefunction doesn't go to zero
    m_numberOfParameters = 3;
    m_parameters.reserve(3);
    m_parameters.push_back(alpha);
    m_parameters.push_back(beta);
    m_parameters.push_back(a);
}

void InteractingGaussian::InitialisePositions(std::vector<std::unique_ptr<class Particle>> &particles)
{
    double r2, dist, u_p;
    assert(particles[0]->getNumberOfDimensions() <= 3);
    m_distances = std::vector<std::vector<double>>();
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        auto pos = particles[i]->getPosition();
        auto temp = std::vector<double>();
        for (unsigned int j = 0; j < i; j++)
        {
            auto pos2 = particles[j]->getPosition();
            r2 = 0;
            for (unsigned int k = 0; k < pos.size(); k++)
            {
                r2 += (pos2[k] - pos[k]) * (pos2[k] - pos[k]);
            }
            temp.push_back(sqrt(r2));
        }
        r2 = 0;
        for (unsigned int k = 0; k < pos.size(); k++)
        {
            r2 += pos[k] * pos[k];
        }
        temp.push_back(r2);
        m_distances.push_back(temp);
    }
    m_interForces = std::vector<double>(3 * particles.size(), 0);
    for (unsigned int k = 0; k < particles.size(); k++)
    {
        auto pos = particles[k]->getPosition();
        for (unsigned int i = 0; i < particles.size(); i++)
        {
            if (i == k)
                continue;
            auto pos2 = particles[i]->getPosition();
            auto relPos = std::vector<double>();
            dist = i < k ? m_distances[k][i] : m_distances[i][k];
            u_p = uPrime_r(dist);
            for (unsigned int j = 0; j < pos.size(); j++)
            {
                m_interForces[3 * k + j] += (pos[j] - pos2[j]) * u_p;
            }
        }
    }
}

void InteractingGaussian::adjustPosition(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> step)
{

    double r2, u_p;
    auto pos_old = particles[index]->getPosition();
    auto pos = particles[index]->getPosition();
    for (unsigned int j = 0; j < pos.size(); j++)
    {
        m_interForces[3 * index + j] = 0;
        pos[j] += step[j];
    }

    for (int j = 0; j < index; j++)
    { // Row
        auto pos2 = particles[j]->getPosition();
        u_p = uPrime_r(m_distances[index][j]);
        for (unsigned int k = 0; k < pos.size(); k++)
            m_interForces[3 * j + k] -= (pos2[k] - pos_old[k]) * u_p;
        r2 = 0;
        for (unsigned int k = 0; k < pos.size(); k++)
        {
            r2 += (pos2[k] - pos[k]) * (pos2[k] - pos[k]);
        }
        m_distances[index][j] = sqrt(r2);
        u_p = uPrime_r(m_distances[index][j]);
        for (unsigned int k = 0; k < pos.size(); k++)
        {
            m_interForces[3 * j + k] += (pos2[k] - pos[k]) * u_p;
            m_interForces[3 * index + k] += (pos[k] - pos2[k]) * u_p;
        }
    }
    r2 = 0;
    for (unsigned int k = 0; k < pos.size(); k++)
    {
        r2 += pos[k] * pos[k];
    }
    m_distances[index][index] = r2;
    for (unsigned int j = index + 1; j < particles.size(); j++)
    { // Column
        auto pos2 = particles[j]->getPosition();
        u_p = uPrime_r(m_distances[j][index]);
        for (unsigned int k = 0; k < pos.size(); k++)
            m_interForces[3 * j + k] -= (pos2[k] - pos_old[k]) * u_p;
        r2 = 0;
        for (unsigned int k = 0; k < pos.size(); k++)
        {
            r2 += (pos2[k] - pos[k]) * (pos2[k] - pos[k]);
        }
        m_distances[j][index] = sqrt(r2);
        u_p = uPrime_r(m_distances[j][index]);
        for (unsigned int k = 0; k < pos.size(); k++)
        {
            m_interForces[3 * j + k] += (pos2[k] - pos[k]) * u_p;
            m_interForces[3 * index + k] += (pos[k] - pos2[k]) * u_p;
        }
    }
}

double InteractingGaussian::uPrime_r(double r)
{
    // *************** [Derivative of log(1.-(m_parameters[2]/r))]/r ***************
    double a = m_parameters[2];
    return a / (r * r * abs(a - r));
}

double InteractingGaussian::uDoublePrime(double r)
{
    // *************** Second derivative of log(1.-(m_parameters[2]/r)) ***************
    double a = m_parameters[2];
    return (a * (a - 2 * r)) / (r * r * (a - r) * (a - r));
    // return 1/(r*r)-1/((m_parameters[2]-r)*(m_parameters[2]-r));
}

double InteractingGaussian::evaluate(std::vector<std::unique_ptr<class Particle>> &particles)
{
    // Returns Phi, not Phi^2
    double r2 = 0, a = m_parameters[2];
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        r2 += m_distances[i][i];
    }
    double phi = exp(-1 * r2 * m_parameters[0]);
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        for (unsigned int j = i + 1; j < particles.size(); j++)
        {
            phi *= std::max(1 - a / m_distances[j][i], 0.);
        }
    }

    return phi;
}

double InteractingGaussian::computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles)
{
    //***************WE RETURN d2/dx2(phi)/phi NOT d2/dx2(phi)*********************
    // The second derivative of exp(-alpha x*x) is exp(-alpha x*x)*(4*alpha*alpha*x*x - 2*alpha)

    // Non-interacting part
    double r2 = 0, alpha = m_parameters[0];
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        r2 += m_distances[i][i];
    }
    int n = particles.size() * particles[0]->getNumberOfDimensions();
    double nabla2 = 4 * alpha * alpha * r2 - 2 * n * alpha;

    // double sum over all particles
    double dist, u_p;
    std::vector<double> repulsion, position;
    for (unsigned int k = 0; k < particles.size(); k++)
    {
        for (unsigned int i = 0; i < k; i++)
        {
            dist = m_distances[k][i];
            u_p = uPrime_r(dist);
            nabla2 += 2 * uDoublePrime(dist) + 4 * u_p;
        }
        position = particles[k]->getPosition();
        for (unsigned int j = 0; j < position.size(); j++)
        {
            // nabla phi = -2*alpha*r
            nabla2 += -4 * alpha * position[j] * m_interForces[3 * k + j];
            nabla2 += m_interForces[3 * k + j] * m_interForces[3 * k + j];
        }
    }

    return nabla2;
}

std::vector<double> InteractingGaussian::quantumForce(std::vector<std::unique_ptr<class Particle>> &particles, int index)
{
    //***************WE RETURN d/dx(phi)/phi NOT d/dx(phi)*********************
    auto pos = particles[index]->getPosition();
    auto force = std::vector<double>(pos);
    double alpha = m_parameters[0];
    for (unsigned int i = 0; i < force.size(); i++)
    {
        force[i] *= -2 * alpha;
        force[i] += m_interForces[3 * index + i];
    }
    return force;
}

std::vector<double> InteractingGaussian::quantumForceMoved(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step)
{
    //***************WE RETURN d/dx(phi)/phi NOT d/dx(phi)*********************

    double r2, temp, alpha = m_parameters[0];
    auto pos = particles[index]->getPosition();
    auto force = std::vector<double>(pos);
    for (unsigned int i = 0; i < force.size(); i++)
    {
        force[i] += step[i];
        force[i] *= -2 * alpha;
    }
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        if ((int)i == index)
            continue;
        auto pos2 = particles[i]->getPosition();
        auto relPos = std::vector<double>();
        r2 = 0;
        for (unsigned int k = 0; k < pos.size(); k++)
        {
            relPos.push_back(pos[k] + step[k] - pos2[k]);
            r2 += relPos[k] * relPos[k];
        }
        temp = sqrt(r2);
        temp = uPrime_r(temp);
        for (unsigned int k = 0; k < pos.size(); k++)
        {
            force[k] += relPos[k] * temp;
        }
    }
    return force;
}

double InteractingGaussian::phiRatio(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step)
{
    // Calculate (phi(new)/phi(old))**2
    double r2 = 0, r2new = 0, diff, Jik, JikNew, a = m_parameters[2];
    auto pos = particles[index]->getPosition();
    for (unsigned int i = 0; i < pos.size(); i++)
        r2 += (pos[i] + step[i]) * (pos[i] + step[i]) - pos[i] * pos[i];
    double phi = exp(-2 * r2 * m_parameters[0]);
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        if ((int)i == index)
            continue;
        auto pos2 = particles[i]->getPosition();
        r2 = 0;
        r2new = 0;
        for (unsigned int k = 0; k < pos.size(); k++)
        {
            diff = pos[k] - pos2[k];
            r2 += diff * diff;
            r2new += (diff + step[k]) * (diff + step[k]);
        }
        Jik = 1 - a / sqrt(r2);
        if (Jik < 0)
            phi *= 1E6; // arbitrary constant to motivate particles that overlap to move
        else
        {
            JikNew = std::max(1 - a / sqrt(r2new), 0.);
            phi *= JikNew * JikNew / (Jik * Jik);
        }
    }
    return phi;
}

std::vector<double> InteractingGaussian::getdPhi_dParams(std::vector<std::unique_ptr<class Particle>> &particles)
{
    double r2 = 0;
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        std::vector<double> position = particles[i]->getPosition();
        for (unsigned int j = 0; j < position.size(); j++)
            r2 += position[j] * position[j];
    }
    return std::vector<double>{-r2, 0};
}