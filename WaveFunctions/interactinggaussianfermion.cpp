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

// Used for the calculation of the determinant
#define N 6

InteractingGaussianFermion::InteractingGaussianFermion(double alpha, double beta, double omega)
{
    assert(alpha > 0); // If alpha == 0 then the wavefunction doesn't go to zero
    m_numberOfParameters = 3;
    m_parameters.reserve(3);
    m_parameters.push_back(alpha);
    m_parameters.push_back(beta);
    m_parameters.push_back(omega);
    m_sqrtAO = sqrt(alpha * omega);
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
            std::cout << std::setw(10) << std::setprecision(3) << dotProduct(array_vals, j);
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
    }
    testInverse(particles);
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

double InteractingGaussianFermion::dotProduct(std::vector<double> &newVals, int index)
{ // Calculates the dot product of a column of the slater matrix with a column of the inverse matrix
    double **invMat = index < m_n_2 ? m_invMatrixUp : m_invMatrixDown;
    int indexReduced = index % m_n_2;
    double sum = 0;
    for (int i = 0; i < m_n_2; i++)
    {
        sum += newVals[i] * invMat[i][indexReduced];
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

// Borrowed from https://www.geeksforgeeks.org/c-program-to-find-determinant-of-a-matrix/ on 09-05-23
void getCofactor(double mat[N][N], double temp[N][N],
                 int p, int q, int n)
{
    int i = 0, j = 0;

    // Looping for each element of the matrix
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            // Copying into temporary matrix
            // only those element which are
            // not in given row and column
            if (row != p && col != q)
            {
                temp[i][j++] = mat[row][col];

                // Row is filled, so increase row
                // index and reset col index
                if (j == n - 1)
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
}

/* Recursive function for finding the
   determinant of matrix. n is current
   dimension of mat[][]. */
// Borrowed from https://www.geeksforgeeks.org/c-program-to-find-determinant-of-a-matrix/ on 09-05-23
double determinantOfMatrix(double mat[N][N], int n)
{
    // Initialize result
    double D = 0;

    //  Base case : if matrix contains
    // single element
    if (n == 1)
        return mat[0][0];

    // To store cofactors
    double temp[N][N];

    // To store sign multiplier
    int sign = 1;

    // Iterate for each element of
    // first row
    for (int f = 0; f < n; f++)
    {
        // Getting Cofactor of mat[0][f]
        getCofactor(mat, temp, 0, f, n);
        D += sign * mat[0][f] * determinantOfMatrix(temp, n - 1);

        // Terms are to be added with alternate sign
        sign = -sign;
    }

    return D;
}

double InteractingGaussianFermion::evaluate(std::vector<std::unique_ptr<class Particle>> &particles)
{
    // assert(particles.size() <= 2 * N);
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
    phi *= determinantOfMatrix(mat, m_n_2);
    for (int i = 0; i < m_n_2; i++)
    {
        for (int j = 0; j < m_n_2; j++)
        {
            mat[i][j] = m_invMatrixDown[i][j];
        }
    }
    phi *= determinantOfMatrix(mat, m_n_2);

    // for (unsigned int i = 0; i < particles.size(); i++)
    // {
    //     for (unsigned int j = i + 1; j < particles.size(); j++)
    //     {
    //         phi *= exp(a*)
    //     }
    // }
    // auto pos1 = particles[0]->getPosition();
    // auto pos2 = particles[1]->getPosition();
    // r2 = (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]);
    // double r = sqrt(r2);
    // phi *= exp(r / (1 + beta * r));
    return phi;
}

double InteractingGaussianFermion::evalPhiPrimePrime(int i, std::vector<double> const &pos, double phi0)
{ // Calculates phi_i given i, r and exp(-alpha*omega*r2)
    int nx = m_nxny[i][0], ny = m_nxny[i][1];
    return ((hermitePrimePrime(nx, pos[0]) * hermite(ny, pos[1])) + (hermite(nx, pos[0]) * hermitePrimePrime(ny, pos[1]))) * phi0;
}

double InteractingGaussianFermion::hermitePrimePrime(int i, double pos)
{ // Calculates the value of the i-th hermite polynomial for a given position
    double rho = m_sqrtAO, rho2, pos2;
    rho2 = rho * rho;
    pos *= rho;
    pos2 = pos * pos;

    // 4*rho2*n*(n - 1)*H(n-2)(pos) - 4*rho3*pos*n*H(n-1)(pos) + (rho4*pos*pos - rho2)H(n)(pos)

    switch (i)
    {
    case 0:
        return (rho2 * pos2 - 1) * rho2;
    case 1:
        return (rho2 * pos2 - rho * 2 - 1) * rho2 * pos * 2;
    case 2:
        return (((-4 * pos2 + 2) * rho2 - 16 * rho + 4) * pos2 + 10) * rho2;
    case 3:
        return (((2 * pos2 - 3) * rho2 - 12 * rho - 2) * pos2 + 6 * rho + 15) * rho2 * pos * 4;
    default:
        assert(false);
    }
}

void InteractingGaussianFermion::arrayValsPrimePrime(std::vector<double> const &pos, std::vector<double> &output)
{ // Evaluates the values is a column of the Slater matrix, given particles position pos and stores the output in output
    // Pos has to be updated before giving it to this function, output has to already be initialised
    double r2 = 0, phi0, AO = m_sqrtAO * m_sqrtAO;
    for (unsigned int i = 0; i < pos.size(); i++)
    {
        r2 += pos[i] * pos[i];
    }
    phi0 = exp(-1 * AO * r2);
    for (int i = 0; i < m_n_2; i++)
    {
        output[i] = evalPhiPrimePrime(i, pos, phi0);
    }
}

double InteractingGaussianFermion::computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles)
{
    //***************WE RETURN d2/dx2(phi)/phi NOT d2/dx2(phi)*********************

    // Non-interacting part
    double nabla2 = 0;
    std::vector<double> array_vals = std::vector<double>(m_n_2, 0), pos;
    for (int i = 0; i < m_n; i++)
    {
        pos = particles[i]->getPosition();
        arrayValsPrimePrime(pos, array_vals);
        nabla2 += dotProduct(array_vals, i);
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
    // double r2 = 0, r2new = 0, r, rnew, diff, Jratio, beta = m_parameters[1];
    // auto pos = particles[index]->getPosition();
    // for (unsigned int i = 0; i < pos.size(); i++)
    //     r2 += step[i] * (2 * pos[i] + step[i]);
    // double phi = exp(-2 * r2 * m_parameters[0]);
    // for (unsigned int i = 0; i < particles.size(); i++)
    // {
    //     if ((int)i == index)
    //         continue;
    //     auto pos2 = particles[i]->getPosition();
    //     r2 = 0;
    //     r2new = 0;
    //     for (unsigned int k = 0; k < pos.size(); k++)
    //     {
    //         diff = pos[k] - pos2[k];
    //         r2 += diff * diff;
    //         r2new += (diff + step[k]) * (diff + step[k]);
    //     }
    //     r = sqrt(r2);
    //     rnew = sqrt(r2new);
    //     Jratio = (rnew / (1 + beta * rnew)) - (r / (1 + beta * r));
    //     phi *= exp(2 * Jratio);
    // }
    // return phi;
    std::vector<double> array_vals = std::vector<double>(m_n_2, 0);
    std::vector<double> pos = particles[index]->getPosition();
    pos[0] += step[0];
    pos[1] += step[1];
    arrayVals(pos, array_vals);
    return dotProduct(array_vals, index);
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