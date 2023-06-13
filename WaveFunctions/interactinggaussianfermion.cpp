#include <assert.h>
#include <iomanip>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <string>

#include "../Math/determinant.h"
#include "../particle.h"
#include "../system.h"
#include "../utils.h"
#include "interactinggaussianfermion.h"
#include "wavefunction.h"

#include "toggles.h"
#include <iostream>

InteractingGaussianFermion::InteractingGaussianFermion(double alpha, double beta, double omega, bool grad_optimisation)
{
    // assert(alpha > 0); // If alpha == 0 then the wavefunction doesn't go to zero
    if (alpha <= 0) alpha = 1E-2;
    // if (beta <= 0) beta = 0;
    m_numberOfParameters = 3;
    m_parameters.reserve(3);
    m_parameters.push_back(alpha);
    m_parameters.push_back(beta);
    m_parameters.push_back(omega);
    m_sqrtAO = sqrt(alpha * omega);
    m_grad_optimisation = grad_optimisation;
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
            std::cout << std::setw(10) << std::setprecision(3) << dotProduct(array_vals, j);
        }
        std::cout << "\t\t";
        for (int j = 0; j < m_n_2; j++)
        {
            std::cout << std::setw(10) << std::setprecision(3) << m_invMatrixUp[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    for (int i = m_n_2; i < m_n; i++)
    {
        arrayVals(particles[i]->getPosition(), array_vals);
        for (int j = m_n_2; j < m_n; j++)
        {
            std::cout << std::setw(10) << std::setprecision(3) << dotProduct(array_vals, j);
        }
        std::cout << "\t\t";
        for (int j = 0; j < m_n_2; j++)
        {
            std::cout << std::setw(10) << std::setprecision(3) << m_invMatrixDown[i - m_n_2][j];
        }
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
    if (m_n > 2 * N)
        std::cout << "Warning: calculating the determinant is not possible" << std::endl;
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
    m_phi_prime = std::vector<std::vector<std::array<double, 2>>>(m_n);
    m_phi_primePrime = std::vector<std::vector<double>>(m_n);
    if (m_grad_optimisation) m_phi_alpha = std::vector<std::vector<double>>(m_n);

    std::vector<double> pos;
    for (int i = 0; i < m_n; i++)
    {
        pos = particles[i]->getPosition();
        arrayVals(pos, array_vals);
        updateInverseMatrix(i, array_vals);
        m_phi_prime[i] = std::vector<std::array<double, 2>>(m_n_2);
        arrayValsPrime(pos, m_phi_prime[i]);
        m_phi_primePrime[i] = std::vector<double>(m_n_2);
        arrayValsPrimePrime(pos, m_phi_primePrime[i]);
        if (m_grad_optimisation)
        {
            m_phi_alpha[i] = std::vector<double>(m_n_2);
            arrayValsAlpha(pos, m_phi_alpha[i]);
        }
    }

#ifdef Interaction
    double r2, r, u_p, a;
    m_distances = std::vector<std::vector<double>>(m_n);
    m_jPrime = std::vector<std::vector<double>>(m_n);
    m_jDoublePrime = std::vector<std::vector<double>>(m_n);
    for (int i = 0; i < m_n; i++)
    {
        auto pos = particles[i]->getPosition();
        auto temp = std::vector<double>(i + 1);
        auto temp2 = std::vector<double>(i);
        auto temp3 = std::vector<double>(i);
        for (int j = 0; j < i; j++)
        {
            auto pos2 = particles[j]->getPosition();
            r2 = 0;
            for (int k = 0; k < m_dim; k++)
            {
                r2 += (pos2[k] - pos[k]) * (pos2[k] - pos[k]);
            }
            r = sqrt(r2);
            a = (i / m_n_2) == (j / m_n_2) ? 3 : 1;
            temp[j] = 1 / r;
            temp2[j] = jPrime(a, r);
            temp3[j] = jDoublePrime(a, r);
        }
        r2 = 0;
        r2 += pos[0] * pos[0];
        r2 += pos[1] * pos[1];
        temp[i] = r2;
        m_distances[i] = temp;
        m_jPrime[i] = temp2;
        m_jDoublePrime[i] = temp3;
    }
    m_interForcesJastrow = std::vector<double>(m_dim * m_n, 0);
    for (int k = 0; k < m_n; k++)
    {
        auto pos = particles[k]->getPosition();
        for (int i = 0; i < k; i++)
        {
            auto pos2 = particles[i]->getPosition();
            u_p = m_jPrime[k][i];
            for (int j = 0; j < m_dim; j++)
            {
                m_interForcesJastrow[m_dim * k + j] += (pos[j] - pos2[j]) * u_p;
                m_interForcesJastrow[m_dim * i + j] -= (pos[j] - pos2[j]) * u_p;
            }
        }
    }
#endif
}

#ifdef CoulombOptimisation
std::vector<std::vector<double>> *InteractingGaussianFermion::getDistances()
{
    return &m_distances;
}
#endif

void InteractingGaussianFermion::adjustPosition(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> step)
{

    std::vector<double> array_vals = std::vector<double>(m_n_2);
    std::vector<double> pos = std::vector<double>(particles[index]->getPosition());
    for (int i = 0; i < m_dim; i++)
    {
        pos[i] += step[i];
    }
    arrayVals(pos, array_vals);
    updateInverseMatrix(index, array_vals);
    arrayValsPrime(pos, m_phi_prime[index]);
    arrayValsPrimePrime(pos, m_phi_primePrime[index]);
    if (m_grad_optimisation) arrayValsAlpha(pos, m_phi_alpha[index]);

#ifdef Interaction
    // Jastrow
    double r2, u_p, r, a;
    std::vector<double> pos_old = particles[index]->getPosition();
    pos = std::vector<double>(pos_old);
    for (int j = 0; j < m_dim; j++)
    {
        m_interForcesJastrow[m_dim * index + j] = 0;
        pos[j] += step[j];
    }

    for (int j = 0; j < index; j++)
    { // Row
        std::vector<double> pos2 = particles[j]->getPosition();
        u_p = m_jPrime[index][j];
        for (int k = 0; k < m_dim; k++)
            m_interForcesJastrow[m_dim * j + k] -= (pos2[k] - pos_old[k]) * u_p;
        r2 = 0;
        for (int k = 0; k < m_dim; k++)
        {
            r2 += (pos2[k] - pos[k]) * (pos2[k] - pos[k]);
        }
        r = sqrt(r2);
        m_distances[index][j] = 1 / r;
        a = (index / m_n_2) == (j / m_n_2) ? 3 : 1;
        u_p = jPrime(a, r);
        m_jPrime[index][j] = u_p;
        m_jDoublePrime[index][j] = jDoublePrime(a, r);
        for (int k = 0; k < m_dim; k++)
        {
            m_interForcesJastrow[m_dim * j + k] += (pos2[k] - pos[k]) * u_p;
            m_interForcesJastrow[m_dim * index + k] += (pos[k] - pos2[k]) * u_p;
        }
    }
    r2 = 0;
    r2 += pos[0] * pos[0];
    r2 += pos[1] * pos[1];
    m_distances[index][index] = r2;
    for (int j = index + 1; j < m_n; j++)
    { // Column
        std::vector<double> pos2 = particles[j]->getPosition();
        u_p = m_jPrime[j][index];
        for (int k = 0; k < m_dim; k++)
            m_interForcesJastrow[m_dim * j + k] -= (pos2[k] - pos_old[k]) * u_p;
        r2 = 0;
        for (int k = 0; k < m_dim; k++)
        {
            r2 += (pos2[k] - pos[k]) * (pos2[k] - pos[k]);
        }
        r = sqrt(r2);
        m_distances[j][index] = 1 / r;
        a = (index / m_n_2) == (j / m_n_2) ? 3 : 1;
        u_p = jPrime(a, r);
        m_jPrime[j][index] = u_p;
        m_jDoublePrime[j][index] = jDoublePrime(a, r);
        for (int k = 0; k < m_dim; k++)
        {
            m_interForcesJastrow[m_dim * j + k] += (pos2[k] - pos[k]) * u_p;
            m_interForcesJastrow[m_dim * index + k] += (pos[k] - pos2[k]) * u_p;
        }
    }
#endif
}

double InteractingGaussianFermion::evalPhi(int i, std::vector<double> const &pos, double phi0)
{ // Calculates phi_i given i, r and exp(-alpha*omega*r2)
    int nx = m_nxny[i][0], ny = m_nxny[i][1];
    return hermite(nx, pos[0]) * hermite(ny, pos[1]) * phi0;
}

std::array<double, 2> InteractingGaussianFermion::evalPhiPrime(int i, std::vector<double> const &pos, double phi0)
{ // Calculates nabla phi_i given i, r and exp(-alpha*omega*r2)
    int nx = m_nxny[i][0], ny = m_nxny[i][1];
    std::array<double, 2> grad = {
        hermitePrime(nx, pos[0]) * hermite(ny, pos[1]) * phi0,
        hermite(nx, pos[0]) * hermitePrime(ny, pos[1]) * phi0};
    return grad;
}

double InteractingGaussianFermion::evalPhiPrimePrime(int i, std::vector<double> const &pos, double phi0)
{ // Calculates nabla^2 phi_i given i, r and exp(-alpha*omega*r2)
    int nx = m_nxny[i][0], ny = m_nxny[i][1];
    return ((hermitePrimePrime(nx, pos[0]) * hermite(ny, pos[1])) + (hermite(nx, pos[0]) * hermitePrimePrime(ny, pos[1]))) * phi0;
}

double InteractingGaussianFermion::evalPhiAlpha(int i, std::vector<double> const &pos, double phi0, double r2)
{ // Calculates nabla^2 phi_i given i, r and exp(-alpha*omega*r2)
    int nx = m_nxny[i][0], ny = m_nxny[i][1];
    double Hx, Hy, Hx_, Hy_, rho = m_sqrtAO, omega = m_parameters[2];
    Hx = hermite(nx, pos[0]);
    Hx_ = hermiteAlpha(nx, pos[0]);
    Hy = hermite(ny, pos[1]);
    Hy_ = hermiteAlpha(ny, pos[1]);
    return ((Hx * Hy_ + Hx_ * Hy) / (rho * 2) - r2 * Hx * Hy) * omega * phi0;
}

double InteractingGaussianFermion::hermite(int i, double pos)
{ // Calculates the value of the i-th hermite polynomial for a given position
    pos *= m_sqrtAO;
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

double InteractingGaussianFermion::hermitePrime(int i, double pos)
{ // Calculates the value of the term in the first derivative of phi_i w.r.t. x
    double rho = m_sqrtAO, pos2;
    pos *= rho;
    pos2 = pos * pos;
    switch (i)
    {
    case 0:
        return -2 * pos * rho;
    case 1:
        return (-4 * pos2 + 2) * rho;
    case 2:
        return (-8 * pos2 + 12) * rho * pos;
    case 3:
        return ((-16 * pos2 + 48) * pos2 - 12) * rho;
    default:
        assert(false);
        return 2 * rho * i * hermite(i - 1, pos / rho) - 2 * rho * pos * hermite(i, pos / rho);
    }
}

double InteractingGaussianFermion::hermitePrimePrime(int i, double pos)
{ // Calculates the value of the term in the second derivative of phi_i w.r.t. x
    double rho = m_sqrtAO, rho2, pos2;
    rho2 = rho * rho;
    pos2 = pos * pos;
    switch (i)
    {
    case 0:
        return (4 * rho2 * pos2 - 2) * rho2;
    case 1:
        return (-12 + 8 * rho2 * pos2) * rho2 * rho * pos;
    case 2:
        return ((16 * rho2 * pos2 - 48) * rho2 * pos2 + 4) * rho2;
    case 3:
        return ((rho2 * pos2 - 5) * 32 * rho2 * pos2 - 24) * rho2 * rho * pos;
    default:
        assert(false);
        return (4 * rho2 * i * (i - 1) * hermite(i - 2, pos)) - (8 * rho2 * rho * pos * i * hermite(i - 1, pos)) + ((4 * rho2 * rho2 * pos2 - 2 * rho2) * hermite(i, pos));
    }
}

double InteractingGaussianFermion::hermiteAlpha(int i, double pos)
{ // Calculates the value of the term in the second derivative of phi_i w.r.t. x
    double rho = m_sqrtAO;
    switch (i)
    {
    case 0:
        return 0;
    case 1:
        return 2 * i * hermite(i - 1, pos) * pos;
        return 2 * pos;
    case 2:
        return 8 * rho * pos * pos;
    case 3:
        return (24 * rho * rho * pos * pos - 12) * pos;
    default:
        assert(false);
        return 2 * i * hermite(i - 1, pos) * pos;
    }
}

void InteractingGaussianFermion::arrayVals(std::vector<double> const &pos, std::vector<double> &output)
{ // Evaluates the values in a column of the Slater matrix, given particles position pos and stores the output in output
    // Pos has to be updated before giving it to this function, output has to already be initialised
    double r2 = 0, phi0, alpha = m_parameters[0], omega = m_parameters[2];
    for (int i = 0; i < m_dim; i++)
    {
        r2 += pos[i] * pos[i];
    }
    phi0 = exp(-alpha * omega * r2);
    for (int i = 0; i < m_n_2; i++)
    {
        output[i] = evalPhi(i, pos, phi0);
    }
}

void InteractingGaussianFermion::arrayValsPrime(std::vector<double> const &pos, std::vector<std::array<double, 2>> &output)
{ // Evaluates the values in a column of the Slater matrix, given particles position pos and stores the output in output
    // Pos has to be updated before giving it to this function, output has to already be initialised
    double r2 = 0, phi0, alpha = m_parameters[0], omega = m_parameters[2];
    for (int i = 0; i < m_dim; i++)
    {
        r2 += pos[i] * pos[i];
    }
    phi0 = exp(-alpha * omega * r2);
    for (int i = 0; i < m_n_2; i++)
    {
        output[i] = evalPhiPrime(i, pos, phi0);
    }
}

void InteractingGaussianFermion::arrayValsPrimePrime(std::vector<double> const &pos, std::vector<double> &output)
{ // Evaluates the values in a column of the Slater matrix, given particles position pos and stores the output in output
    // Pos has to be updated before giving it to this function, output has to already be initialised
    double r2 = 0, phi0, AO = m_sqrtAO * m_sqrtAO;
    for (int i = 0; i < m_dim; i++)
    {
        r2 += pos[i] * pos[i];
    }
    phi0 = exp(-1 * AO * r2);
    for (int i = 0; i < m_n_2; i++)
    {
        output[i] = evalPhiPrimePrime(i, pos, phi0);
    }
}

void InteractingGaussianFermion::arrayValsAlpha(std::vector<double> const &pos, std::vector<double> &output)
{ // Evaluates the values in a column of the Slater matrix, given particles position pos and stores the output in output
    // Pos has to be updated before giving it to this function, output has to already be initialised
    double r2 = 0, phi0, AO = m_sqrtAO * m_sqrtAO;
    for (int i = 0; i < m_dim; i++)
    {
        r2 += pos[i] * pos[i];
    }
    phi0 = exp(-1 * AO * r2);
    for (int i = 0; i < m_n_2; i++)
    {
        output[i] = evalPhiAlpha(i, pos, phi0, r2);
    }
}

double InteractingGaussianFermion::dotProduct(std::vector<double> &vals, int index)
{ // Calculates the dot product of a column of the slater matrix with a column of the inverse matrix
    double **invMat = index < m_n_2 ? m_invMatrixUp : m_invMatrixDown;
    int indexReduced = index % m_n_2;
    double sum = 0;
    for (int i = 0; i < m_n_2; i++)
    {
        sum += vals[i] * invMat[i][indexReduced];
    }
    return sum;
}

std::vector<double> InteractingGaussianFermion::vectorDotProduct(std::vector<std::array<double, 2>> &vals, int index)
{ // Calculates the dot product of a column of the slater matrix with a column of the inverse matrix
    double **invMat = index < m_n_2 ? m_invMatrixUp : m_invMatrixDown;
    int indexReduced = index % m_n_2;
    std::vector<double> sum = std::vector<double>(2, 0);
    for (int i = 0; i < m_n_2; i++)
    {
        sum[0] += vals[i][0] * invMat[i][indexReduced];
        sum[1] += vals[i][1] * invMat[i][indexReduced];
    }
    return sum;
}

void InteractingGaussianFermion::updateInverseMatrix(int index, std::vector<double> arrayVals)
{
    double **invMat = index < m_n_2 ? m_invMatrixUp : m_invMatrixDown;
    int indexReduced = index % m_n_2;
    int offset = index < m_n_2 ? 0 : m_n_2;
    double R_, S_R;
    R_ = 1 / dotProduct(arrayVals, index);
    for (int i = 0; i < m_n_2; i++)
    {
        if (i == indexReduced) continue;
        S_R = dotProduct(arrayVals, i + offset) * R_;
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

double InteractingGaussianFermion::jPrime(double a, double r)
{
    // *************** 1/(rij*(1+beta*rij)^2) ***************
    double beta = m_parameters[1], betaTerm;
    betaTerm = 1 + beta * r;
    return 1 / (a * r * betaTerm * betaTerm);
}

double InteractingGaussianFermion::jDoublePrime(double a, double r)
{
    // *************** Second derivative  ***************
    double beta = m_parameters[1], betaTerm;
    betaTerm = 1 + beta * r;
    return (1 - beta * r) / (a * r * betaTerm * betaTerm * betaTerm);
    ;
}

double InteractingGaussianFermion::evaluate(std::vector<std::unique_ptr<class Particle>> &particles)
{
    assert(particles.size() <= 2 * N);
    assert(particles[0]->getNumberOfDimensions() == 2);

    double phi = 1;
    double mat[N][N];
    for (int i = 0; i < m_n_2; i++)
    {
        for (int j = 0; j < m_n_2; j++)
        {
            mat[i][j] = m_invMatrixUp[i][j];
        }
    }
    phi /= determinantOfMatrix(mat, m_n_2);
    for (int i = 0; i < m_n_2; i++)
    {
        for (int j = 0; j < m_n_2; j++)
        {
            mat[i][j] = m_invMatrixDown[i][j];
        }
    }
    phi /= determinantOfMatrix(mat, m_n_2);

// testInverse(particles);
#ifdef Interaction
    std::vector<double> pos1, pos2;
    double r2, r, a, beta = m_parameters[1];
    for (int i = 0; i < m_n; i++)
    {
        pos1 = particles[i]->getPosition();
        for (int j = 0; j < i; j++)
        {
            pos2 = particles[j]->getPosition();

            r2 = (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]);
            r = sqrt(r2);
            a = (i / m_n_2) == (j / m_n_2) ? 3 : 1; // We divide by 1/a instead of multiplying by a, because 3 is easier to use than 1/3
            phi *= exp(r / (a * (1 + beta * r)));
        }
    }
#endif
    return phi;
}

double InteractingGaussianFermion::computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles)
{
    //***************WE RETURN d2/dx2(phi)/phi NOT d2/dx2(phi)*********************
    // Non-interacting part
    double nabla2 = 0;
    for (int i = 0; i < m_n; i++)
    {
        nabla2 += dotProduct(m_phi_primePrime[i], i);
    }
#ifdef Interaction
    std::vector<double> force, interForce;
    for (int i = 0; i < m_n; i++)
    {
        force = quantumForceSlater(i);
        interForce = quantumForceJastrow(particles, i);
        for (int j = 0; j < m_dim; j++)
        {
            nabla2 += (2 * force[j] + interForce[j]) * interForce[j];
        }
    }

    for (int i = 0; i < m_n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            nabla2 += m_dim * m_jDoublePrime[i][j];
        }
    }
#endif

    return nabla2;
}

std::vector<double> InteractingGaussianFermion::quantumForceSlater(int index)
{
    return vectorDotProduct(m_phi_prime[index], index);
}

std::vector<double> InteractingGaussianFermion::quantumForceJastrow(std::vector<std::unique_ptr<class Particle>> &, int index)
{
    return std::vector<double>{m_interForcesJastrow[m_dim * index], m_interForcesJastrow[m_dim * index + 1]};
}

std::vector<double> InteractingGaussianFermion::quantumForce(std::vector<std::unique_ptr<class Particle>> &particles, int index)
{
    //***************WE RETURN d/dx(phi)/phi NOT d/dx(phi)*********************
    std::vector<double> force = quantumForceSlater(index);
#ifdef Interaction
    std::vector<double> interForce = quantumForceJastrow(particles, index);
    for (int i = 0; i < m_dim; i++)
    {
        force[i] += interForce[i];
    }
#endif
    return force;
}

std::vector<double> InteractingGaussianFermion::quantumForceMoved(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step)
{
    //***************WE RETURN d/dx(phi)/phi NOT d/dx(phi)*********************
    // The inverse matrix should be updated before calculating the gradient like in the quantumForce
    // However, only one row of the updated matix will be used
    // Updating the matrix results in dividing the row by the determinent ratio
    // Instead we divide by the determinant ratio here
    std::vector<double> pos = std::vector<double>(particles[index]->getPosition());
    pos[0] += step[0];
    pos[1] += step[1];
    std::vector<std::array<double, 2>> array_vecs = std::vector<std::array<double, 2>>(m_n_2);
    arrayValsPrime(pos, array_vecs);
    std::vector<double> force = vectorDotProduct(array_vecs, index);
    std::vector<double> array_vals = std::vector<double>(m_n_2);
    arrayVals(pos, array_vals);
    double det_ratio = dotProduct(array_vals, index);
    force[0] /= det_ratio;
    force[1] /= det_ratio;

#ifdef Interaction
    double a, r2, r, j_p;
    std::vector<double> pos1;
    for (int i = 0; i < m_n; i++)
    {
        if (i == index) continue;
        pos1 = particles[i]->getPosition();
        a = (i / m_n_2) == (index / m_n_2) ? 3 : 1; // We divide by 1/a instead of multiplying by a, because 3 is easier to use than 1/3
        r2 = (pos[0] - pos1[0]) * (pos[0] - pos1[0]) + (pos[1] - pos1[1]) * (pos[1] - pos1[1]);
        r = sqrt(r2);
        j_p = jPrime(a, r);
        force[0] += (pos[0] - pos1[0]) * j_p;
        force[1] += (pos[1] - pos1[1]) * j_p;
    }
#endif

    return force;
}

double InteractingGaussianFermion::phiRatio(std::vector<std::unique_ptr<class Particle>> &particles, int index, std::vector<double> &step)
{
    // Calculate (phi(new)/phi(old))**2
    std::vector<double> array_vals = std::vector<double>(m_n_2, 0);
    std::vector<double> pos = particles[index]->getPosition();
    std::vector<double> pos_new = std::vector<double>(2);
    pos_new[0] = pos[0] + step[0];
    pos_new[1] = pos[1] + step[1];
    arrayVals(pos_new, array_vals);
    double ratio = dotProduct(array_vals, index);

#ifdef Interaction
    double a, r2, r, r_new, beta = m_parameters[1], sum = 0;
    std::vector<double> pos1;
    for (int i = 0; i < m_n; i++)
    {
        if (i == index) continue;
        pos1 = particles[i]->getPosition();
        a = (i / m_n_2) == (index / m_n_2) ? 3 : 1; // We divide by 1/a instead of multiplying by a, because 3 is easier to use than 1/3
        r2 = (pos[0] - pos1[0]) * (pos[0] - pos1[0]) + (pos[1] - pos1[1]) * (pos[1] - pos1[1]);
        r = sqrt(r2);
        r2 = (pos_new[0] - pos1[0]) * (pos_new[0] - pos1[0]) + (pos_new[1] - pos1[1]) * (pos_new[1] - pos1[1]);
        r_new = sqrt(r2);
        sum += r_new / (a * (1 + beta * r_new)) - r / (a * (1 + beta * r));
    }
    ratio *= exp(sum);
#endif

    return ratio * ratio;
}

void InteractingGaussianFermion::changeAlpha(std::vector<std::unique_ptr<class Particle>> &particles)
{
    std::vector<double> array_vals = std::vector<double>(m_n_2);
    for (int i = 0; i < m_n; i++)
    {
        arrayVals(particles[i]->getPosition(), array_vals);
        updateInverseMatrix(i, array_vals);
    }
}

std::vector<double> InteractingGaussianFermion::getdPhi_dParams(std::vector<std::unique_ptr<class Particle>> &particles)
{
    // Non-interacting part
    double ddAlpha = 0, ddBeta = 0;
    std::vector<double> array_vals = std::vector<double>(m_n_2, 0), pos;
    for (int i = 0; i < m_n; i++)
    {
        if (m_grad_optimisation)
        {
            ddAlpha += dotProduct(m_phi_alpha[i], i);
        }
        else
        {
            pos = particles[i]->getPosition();
            arrayValsAlpha(pos, array_vals);
            ddAlpha += dotProduct(array_vals, i);
        }
    }

#ifdef TestDDalpha
    // Numerical calculation
    double phi, phi_plus, phi_minus, grad = 0, alpha = m_parameters[0], omega = m_parameters[2];
    const double dalpha = 1e-5, dalpha2 = 2 * dalpha;
    phi = evaluate(particles);
    m_parameters[0] = alpha + dalpha;
    m_sqrtAO = sqrt((alpha + dalpha) * omega);
    changeAlpha(particles);
    phi_plus = evaluate(particles);
    m_parameters[0] = alpha - dalpha;
    m_sqrtAO = sqrt((alpha - dalpha) * omega);
    changeAlpha(particles);
    phi_minus = evaluate(particles);
    m_parameters[0] = alpha;
    m_sqrtAO = sqrt(alpha * omega);
    changeAlpha(particles);

    grad = (phi_plus - phi_minus) / (dalpha2 * phi);
    std::cout << "ddAlpha: " << ddAlpha << "\tddAlpha Diff: " << grad - ddAlpha << std::endl;

#endif
#ifdef Interaction
    std::vector<double> pos1, pos2;
    double r2, r, br1, a, beta = m_parameters[1];
    for (int i = 0; i < m_n; i++)
    {
        pos1 = particles[i]->getPosition();
        for (int j = 0; j < i; j++)
        {
            pos2 = particles[j]->getPosition();

            r2 = (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]);
            r = sqrt(r2);
            br1 = beta * r + 1;
            a = (i / m_n_2) == (j / m_n_2) ? 3 : 1; // We divide by 1/a instead of multiplying by a, because 3 is easier to use than 1/3
            // We assume that the number of dimensions = 2
            ddBeta -= r2 / (a * br1 * br1);
        }
    }
#ifdef TestDDbeta
// Numerical calculation
#ifdef TestDDalpha
#else
    double phi, phi_plus, phi_minus, grad = 0;
#endif
    const double dbeta = 1e-5, dbeta2 = 2 * dbeta;
    phi = evaluate(particles);
    m_parameters[1] = beta + dbeta;
    phi_plus = evaluate(particles);
    m_parameters[1] = beta - dbeta;
    phi_minus = evaluate(particles);
    m_parameters[1] = beta;

    grad = (phi_plus - phi_minus) / (dbeta2 * phi);
    std::cout << "ddBeta: " << ddBeta << "\tddBeta Diff: " << grad - ddBeta << std::endl;
#endif

#endif
    return std::vector<double>{ddAlpha, ddBeta};
}