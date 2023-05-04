#include <assert.h>
#include <iomanip>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <string>

#include "../particle.h"
#include "../system.h"
#include "../utils.h"
#include "interactinggaussianfermion.h"
#include "wavefunction.h"

#include <iostream>

InteractingGaussianFermion::InteractingGaussianFermion(double alpha, double beta, double omega)
{
    assert(alpha > 0); // If alpha == 0 then the wavefunction doesn't go to zero
    m_numberOfParameters = 3;
    m_parameters.reserve(3);
    m_parameters.push_back(alpha);
    m_parameters.push_back(beta);
    m_parameters.push_back(omega);
    m_sqrtOmega = sqrt(omega);
}

InteractingGaussianFermion::~InteractingGaussianFermion()
{
    delete_2d_array<double>(m_invMatrixUp);
    delete_2d_array<double>(m_invMatrixDown);
}

void InteractingGaussianFermion::testInverse(std::vector<std::unique_ptr<class Particle>> &particles)
{
    std::vector<double> array_vals = std::vector<double>(m_n_2);
    for (int i = 0; i < m_n_2; i++)
    {
        arrayVals(particles[i]->getPosition(), array_vals);
        for (int j = 0; j < m_n_2; j++)
        {
            std::cout << std::setw(10) << std::setprecision(3) << dotProduct(array_vals, j, m_invMatrixUp);
        }
        std::cout << "\t\t";
        std::cout << std::setprecision(3) << m_invMatrixUp[i][0] << '\t';
        std::cout << std::setprecision(3) << m_invMatrixUp[i][1] << '\t';
        std::cout << std::setprecision(3) << m_invMatrixUp[i][2] << '\t';
        std::cout << std::endl;
    }
    std::cout << std::endl;
    for (int i = m_n_2; i < m_n; i++)
    {
        arrayVals(particles[i]->getPosition(), array_vals);
        for (int j = m_n_2; j < m_n; j++)
        {
            std::cout << std::setw(10) << std::setprecision(3) << dotProduct(array_vals, j, m_invMatrixDown);
        }
        std::cout << "\t\t";
        std::cout << std::setprecision(3) << m_invMatrixDown[i - m_n_2][0] << '\t';
        std::cout << std::setprecision(3) << m_invMatrixDown[i - m_n_2][1] << '\t';
        std::cout << std::setprecision(3) << m_invMatrixDown[i - m_n_2][2] << '\t';
        std::cout << std::endl;
    }
    std::cout << std::endl
              << std::endl;
}

void InteractingGaussianFermion::InitialisePositions(std::vector<std::unique_ptr<class Particle>> &particles)
{
    m_n = (int)particles.size();
    assert(m_n == 2 || m_n == 6 || m_n == 12 || m_n == 20);
    assert(particles[0]->getNumberOfDimensions() == 2);
    m_n_2 = m_n / 2;
    m_nxny = std::vector<std::array<int, 2>>();
    m_nxny.reserve(m_n_2);
    int nx = 0, ny = 0, nxy = 0, i = 0;
    while (i < m_n_2)
    {
        m_nxny[i] = std::array<int, 2>();
        m_nxny[i][0] = nx;
        m_nxny[i][1] = ny;
        if (nx > 0)
        {
            nx--;
            ny++;
        }
        else
        {
            nxy++;
            nx = nxy;
            ny = 0;
        }
        i++;
    }

    m_invMatrixUp = init_2d_array<double>(m_n_2, m_n_2, 0);
    m_invMatrixDown = init_2d_array<double>(m_n_2, m_n_2, 0);
    for (int i = 0; i < m_n_2; i++)
    {
        m_invMatrixUp[i][i] = 1;
        m_invMatrixDown[i][i] = 1;
    }
    std::vector<double> array_vals = std::vector<double>(m_n_2);
    for (int i = 0; i < m_n; i++)
    {
        arrayVals(particles[i]->getPosition(), array_vals);
        updateInverseMatrix(i, array_vals);
        testInverse(particles);
    }

    // testInverse(particles);
}

void InteractingGaussianFermion::adjustPosition(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> step)
{

    std::vector<double> array_vals = std::vector<double>(m_n);
    std::vector<double> pos = std::vector<double>(particles[index]->getPosition());
    for (unsigned int i = 0; i < pos.size(); i++)
    {
        pos[i] += step[i];
    }
    arrayVals(pos, array_vals);
    updateInverseMatrix(index, array_vals);
}

double InteractingGaussianFermion::evalPhi(int i, std::vector<double> const &pos, double phi0)
{ // Calculates phi_i given i, r and exp(-alpha*omega*r2)
    int nx = m_nxny[i][0], ny = m_nxny[i][1];
    return hermite(nx, pos[0]) * hermite(ny, pos[1]) * phi0;
}

double InteractingGaussianFermion::hermite(int i, double pos)
{ // Calculates the value of the i-th hermite polynomial for a given position
    pos *= m_sqrtOmega;
    switch (i)
    {
    case 0:
        return 1;
    case 1:
        return 2 * pos;
    case 2:
        return 4 * pos * pos - 2;
    case 3:
        return pos * (8 * pos * pos - 12);
    default:
        assert(false);
    }
}

void InteractingGaussianFermion::arrayVals(std::vector<double> const &pos, std::vector<double> &output)
{ // Evaluates the values is a column of the Slater matrix, given particles position pos and stores the output in output
    // Pos has to be updated before giving it to this function, output has to already be initialised
    double r2 = 0, phi0, alpha = m_parameters[0], omega = m_parameters[2];
    for (unsigned int i = 0; i < pos.size(); i++)
    {
        r2 += pos[i] * pos[i];
    }
    phi0 = exp(-alpha * omega * r2);
    for (int i = 0; i < m_n_2; i++)
    {
        output[i] = evalPhi(i, pos, phi0);
    }
}

double InteractingGaussianFermion::dotProduct(std::vector<double> &newVals, int invMatIndex, double **invMat)
{ // Calculates the dot product of a column of the slater matrix with a column of the inverse matrix
    double sum = 0;
    for (int i = 0; i < m_n_2; i++)
    {
        sum += newVals[i] * invMat[i][invMatIndex];
    }
    return sum;
}

void InteractingGaussianFermion::updateInverseMatrix(int index, std::vector<double> arrayVals)
{
    double **invMat = index < m_n_2 ? m_invMatrixUp : m_invMatrixDown;
    int indexReduced = index % m_n_2;
    double R_, S_R;
    R_ = 1 / dotProduct(arrayVals, indexReduced, invMat);
    for (int i = 0; i < m_n_2; i++)
    {
        if (i == indexReduced) continue;
        S_R = dotProduct(arrayVals, i, invMat) * R_;
        for (int j = 0; j < m_n_2; j++)
        {
            invMat[j][i] -= S_R * invMat[j][indexReduced];
        }
    }
    for (int j = 0; j < m_n_2; j++)
    {
        invMat[j][indexReduced] *= R_;
    }
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