#include <assert.h>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <string>

#include "../particle.h"
#include "../system.h"
#include "interactinggaussianfermion.h"
#include "wavefunction.h"

#include <iostream>

InteractingGaussianFermion::InteractingGaussianFermion(double alpha, double beta, double)
{
    assert(alpha > 0); // If alpha == 0 then the wavefunction doesn't go to zero
    m_numberOfParameters = 2;
    m_parameters.reserve(2);
    m_parameters.push_back(alpha);
    m_parameters.push_back(beta);
}

void InteractingGaussianFermion::InitialisePositions(std::vector<std::unique_ptr<class Particle>> &particles)
{
    // double r2, dist, u_p;
    // assert(particles[0]->getNumberOfDimensions() <= 3);
    // m_distances = std::vector<std::vector<double>>();
    // for (unsigned int i = 0; i < particles.size(); i++)
    // {
    //     auto pos = particles[i]->getPosition();
    //     auto temp = std::vector<double>();
    //     for (unsigned int j = 0; j < i; j++)
    //     {
    //         auto pos2 = particles[j]->getPosition();
    //         r2 = 0;
    //         for (unsigned int k = 0; k < pos.size(); k++)
    //         {
    //             r2 += (pos2[k] - pos[k]) * (pos2[k] - pos[k]);
    //         }
    //         temp.push_back(sqrt(r2));
    //     }
    //     r2 = 0;
    //     for (unsigned int k = 0; k < pos.size(); k++)
    //     {
    //         r2 += pos[k] * pos[k];
    //     }
    //     temp.push_back(r2);
    //     m_distances.push_back(temp);
    // }
    // m_interForces = std::vector<double>(3 * particles.size(), 0);
    // for (unsigned int k = 0; k < particles.size(); k++)
    // {
    //     auto pos = particles[k]->getPosition();
    //     for (unsigned int i = 0; i < particles.size(); i++)
    //     {
    //         if (i == k)
    //             continue;
    //         auto pos2 = particles[i]->getPosition();
    //         auto relPos = std::vector<double>();
    //         dist = i < k ? m_distances[k][i] : m_distances[i][k];
    //         u_p = uPrime_r(dist);
    //         for (unsigned int j = 0; j < pos.size(); j++)
    //         {
    //             m_interForces[3 * k + j] += (pos[j] - pos2[j]) * u_p;
    //         }
    //     }
    // }
}

void InteractingGaussianFermion::adjustPosition(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> step)
{

    // double r2, u_p;
    // auto pos_old = particles[index]->getPosition();
    // auto pos = particles[index]->getPosition();
    // for (unsigned int j = 0; j < pos.size(); j++)
    // {
    //     m_interForces[3 * index + j] = 0;
    //     pos[j] += step[j];
    // }

    // for (int j = 0; j < index; j++)
    // { // Row
    //     auto pos2 = particles[j]->getPosition();
    //     u_p = uPrime_r(m_distances[index][j]);
    //     for (unsigned int k = 0; k < pos.size(); k++)
    //         m_interForces[3 * j + k] -= (pos2[k] - pos_old[k]) * u_p;
    //     r2 = 0;
    //     for (unsigned int k = 0; k < pos.size(); k++)
    //     {
    //         r2 += (pos2[k] - pos[k]) * (pos2[k] - pos[k]);
    //     }
    //     m_distances[index][j] = sqrt(r2);
    //     u_p = uPrime_r(m_distances[index][j]);
    //     for (unsigned int k = 0; k < pos.size(); k++)
    //     {
    //         m_interForces[3 * j + k] += (pos2[k] - pos[k]) * u_p;
    //         m_interForces[3 * index + k] += (pos[k] - pos2[k]) * u_p;
    //     }
    // }
    // r2 = 0;
    // for (unsigned int k = 0; k < pos.size(); k++)
    // {
    //     r2 += pos[k] * pos[k];
    // }
    // m_distances[index][index] = r2;
    // for (unsigned int j = index + 1; j < particles.size(); j++)
    // { // Column
    //     auto pos2 = particles[j]->getPosition();
    //     u_p = uPrime_r(m_distances[j][index]);
    //     for (unsigned int k = 0; k < pos.size(); k++)
    //         m_interForces[3 * j + k] -= (pos2[k] - pos_old[k]) * u_p;
    //     r2 = 0;
    //     for (unsigned int k = 0; k < pos.size(); k++)
    //     {
    //         r2 += (pos2[k] - pos[k]) * (pos2[k] - pos[k]);
    //     }
    //     m_distances[j][index] = sqrt(r2);
    //     u_p = uPrime_r(m_distances[j][index]);
    //     for (unsigned int k = 0; k < pos.size(); k++)
    //     {
    //         m_interForces[3 * j + k] += (pos2[k] - pos[k]) * u_p;
    //         m_interForces[3 * index + k] += (pos[k] - pos2[k]) * u_p;
    //     }
    // }
}

double InteractingGaussianFermion::jPrime(double r)
{
    // *************** 1/(rij*(1+beta*rij)^2) ***************
    double beta = m_parameters[1], betaTerm, betaTerm2;
    betaTerm = 1 + beta * r;
    betaTerm2 = betaTerm * betaTerm;
    return 1 / (r * betaTerm2);
}

double InteractingGaussianFermion::jDoublePrime(double a, double r)
{
    // *************** Second derivative  ***************
    double beta = m_parameters[1], betaTerm, betaTerm2, betaTerm4;
    betaTerm = 1 + beta * r;
    betaTerm2 = betaTerm * betaTerm;
    betaTerm4 = betaTerm2 * betaTerm2;
    return (a * a * r + a * (1 - 1 * beta * r) * betaTerm) / (betaTerm4 * r);
}

double InteractingGaussianFermion::evaluate(std::vector<std::unique_ptr<class Particle>> &particles)
{
    assert(particles.size() == 2);
    assert(particles[0]->getNumberOfDimensions() == 2);
    // Returns Phi, not Phi^2
    double r2 = 0, alpha = m_parameters[0], beta = m_parameters[1];
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        auto pos = particles[i]->getPosition();
        for (unsigned int j = 0; j < pos.size(); j++)
        {
            r2 += pos[j] * pos[j];
        }
    }
    double phi = exp(-1 * alpha * r2);
    // for (unsigned int i = 0; i < particles.size(); i++)
    // {
    //     for (unsigned int j = i + 1; j < particles.size(); j++)
    //     {
    //         phi *= exp(a*)
    //     }
    // }
    auto pos1 = particles[0]->getPosition();
    auto pos2 = particles[1]->getPosition();
    r2 = (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]);
    double r = sqrt(r2);
    phi *= exp(r / (1 + beta * r));
    return phi;
}

double InteractingGaussianFermion::computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles)
{
    //***************WE RETURN d2/dx2(phi)/phi NOT d2/dx2(phi)*********************
    // The second derivative of exp(-alpha x*x) is exp(-alpha x*x)*(4*alpha*alpha*x*x - 2*alpha)

    // Non-interacting part
    double r2 = 0, alpha = m_parameters[0];
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        auto pos = particles[i]->getPosition();
        for (unsigned int j = 0; j < pos.size(); j++)
        {
            r2 += pos[j] * pos[j];
        }
    }
    int n = particles.size() * particles[0]->getNumberOfDimensions();
    double nabla2 = 4 * alpha * alpha * r2 - 2 * n * alpha;

    // double sum over all particles
    double dist, j_p;
    std::vector<double> repulsion, position;
    for (unsigned int k = 0; k < particles.size(); k++)
    {
        auto pos = particles[k]->getPosition();
        for (unsigned int i = 0; i < k; i++)
        {
            auto pos2 = particles[i]->getPosition();
            r2 = 0;
            for (unsigned int j = 0; j < pos.size(); j++)
            {
                r2 += (pos[j] - pos2[j]) * (pos[j] - pos2[j]);
            }

            dist = sqrt(r2);
            j_p = jPrime(dist);
            nabla2 += 2 * jDoublePrime(1, dist);

            for (unsigned int j = 0; j < pos.size(); j++)
            {
                nabla2 += 2 * (-2 * alpha * (pos[j] - pos2[j])) * (1 * (pos[j] - pos2[j]) * j_p);
            }
        }
    }

    return nabla2;
}

std::vector<double> InteractingGaussianFermion::quantumForce(std::vector<std::unique_ptr<class Particle>> &particles, int index)
{
    //***************WE RETURN d/dx(phi)/phi NOT d/dx(phi)*********************
    auto pos = particles[index]->getPosition();
    auto force = std::vector<double>(pos);
    double alpha = m_parameters[0], r2, j_p;
    for (unsigned int i = 0; i < force.size(); i++)
    {
        force[i] *= -2 * alpha;
    }
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        if ((int)i == index) continue;
        auto pos2 = particles[i]->getPosition();
        r2 = 0;
        for (unsigned int j = 0; j < force.size(); j++)
        {
            r2 += (pos[j] - pos2[j]) * (pos[j] - pos2[j]);
        }
        j_p = jPrime(sqrt(r2));
        for (unsigned int j = 0; j < force.size(); j++)
        {
            force[j] += (pos[j] - pos2[j]) * j_p;
        }
    }
    return force;
}

std::vector<double> InteractingGaussianFermion::quantumForceMoved(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step)
{
    //***************WE RETURN d/dx(phi)/phi NOT d/dx(phi)*********************

    auto pos = particles[index]->getPosition();
    auto force = std::vector<double>(pos);
    double alpha = m_parameters[0], r2 = 0, j_p;
    for (unsigned int i = 0; i < force.size(); i++)
    {
        force[i] += step[i];
        force[i] *= -2 * alpha;
    }
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        if ((int)i == index) continue;
        auto pos2 = particles[i]->getPosition();
        r2 = 0;
        for (unsigned int j = 0; j < force.size(); j++)
        {
            r2 += (pos[j] + step[j] - pos2[j]) * (pos[j] + step[j] - pos2[j]);
        }
        j_p = jPrime(sqrt(r2));
        for (unsigned int j = 0; j < force.size(); j++)
        {
            force[j] += (pos[j] + step[j] - pos2[j]) * j_p;
        }
    }
    return force;
}

double InteractingGaussianFermion::phiRatio(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step)
{
    // Calculate (phi(new)/phi(old))**2
    double r2 = 0, r2new = 0, r, rnew, diff, Jratio, beta = m_parameters[1];
    auto pos = particles[index]->getPosition();
    for (unsigned int i = 0; i < pos.size(); i++)
        r2 += step[i] * (2 * pos[i] + step[i]);
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
        r = sqrt(r2);
        rnew = sqrt(r2new);
        Jratio = (rnew / (1 + beta * rnew)) - (r / (1 + beta * r));
        phi *= exp(2 * Jratio);
    }
    return phi;
}

std::vector<double> InteractingGaussianFermion::getdPhi_dParams(std::vector<std::unique_ptr<class Particle>> &particles)
{
    double r2 = 0;
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        std::vector<double> position = particles[i]->getPosition();
        for (unsigned int j = 0; j < position.size(); j++)
            r2 += position[j] * position[j];
    }
    double r2_ = 0, r, beta = m_parameters[1];
    std::vector<double> pos1, pos2;
    pos1 = particles[0]->getPosition();
    pos2 = particles[1]->getPosition();
    r2_ = (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]);
    r = sqrt(r2_);

    return std::vector<double>{-r2, -r2_ / ((1 + beta * r) * (1 + beta * r))};
}