#include <memory>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <string>

#include "interactinggaussian3d.h"
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"

#include <iostream>


InteractingGaussian3D::InteractingGaussian3D(double alpha, double beta, double a)
{
    assert(alpha > 0); // If alpha == 0 then the wavefunction doesn't go to zero
    m_numberOfParameters = 3;
    m_parameters.reserve(3);
    m_parameters.push_back(alpha);
    m_parameters.push_back(beta);
    m_parameters.push_back(a);
}

void InteractingGaussian3D::InitialisePositions(std::vector<std::unique_ptr<class Particle>>& particles){
    double r2, r, u_p, beta = m_parameters[1];
    assert(particles[0]->getNumberOfDimensions() == 3);
    m_distances = std::vector<std::vector<double>>();
    m_uPrime = std::vector<std::vector<double>>();
    m_uDoublePrime = std::vector<std::vector<double>>();
    for (unsigned int i = 0; i < particles.size(); i++){
        auto pos = particles[i]->getPosition();
        auto temp = std::vector<double>();
        auto temp2 = std::vector<double>();
        auto temp3 = std::vector<double>();
        for (unsigned int j = 0; j<i; j++){
            auto pos2 = particles[j]->getPosition();
            r2 = 0;
            for (unsigned int k = 0; k<3; k++){
                r2 += (pos2[k] - pos[k])*(pos2[k] - pos[k]);
            }
            r = sqrt(r2);
            temp.push_back(r);
            temp2.push_back(uPrime_r(r));
            temp3.push_back(uDoublePrime(r));
        }
        r2 = 0;
        r2 += pos[0] * pos[0];
        r2 += pos[1] * pos[1];
        r2 += pos[2] * pos[2] * beta * beta;
        temp.push_back(r2);
        m_distances.push_back(temp);
        m_uPrime.push_back(temp2);
        m_uDoublePrime.push_back(temp3);
    }
    m_interForces = std::vector<double>(3*particles.size(),0);
    for (unsigned int k = 0; k < particles.size(); k++){
        auto pos = particles[k]->getPosition();
        for (unsigned int i = 0; i<particles.size(); i++){
            if (i == k) continue;
            auto pos2 = particles[i]->getPosition();
            u_p = i<k? m_uPrime[k][i] : m_uPrime[i][k];
            for (unsigned int j = 0; j<3; j++){
                m_interForces[3*k+j] += (pos[j]-pos2[j]) * u_p;
            }
        }
    }
}

void InteractingGaussian3D::adjustPosition(std::vector<std::unique_ptr<class Particle>>& particles, int index, std::vector<double> step){
    
    double r2, u_p, r;
    auto pos_old = particles[index]->getPosition();
    auto pos = particles[index]->getPosition();
    for (unsigned int j = 0; j<3; j++){
        m_interForces[3*index+j] = 0;
        pos[j] += step[j];
    }

    for (int j = 0; j<index; j++){ // Row
        auto pos2 = particles[j]->getPosition();
        u_p = m_uPrime[index][j];
        for (unsigned int k = 0; k<3; k++)
            m_interForces[3*j+k] -= (pos2[k] - pos_old[k])*u_p;
        r2 = 0;
        for (unsigned int k = 0; k<3; k++){
            r2 += (pos2[k] - pos[k])*(pos2[k] - pos[k]);
        }
        r = sqrt(r2);
        m_distances[index][j]=r;
        u_p = uPrime_r(r);
        m_uPrime[index][j]=u_p;
        m_uDoublePrime[index][j]=uDoublePrime(r);
        for (unsigned int k = 0; k<3; k++){
            m_interForces[3*j+k] += (pos2[k] - pos[k])*u_p;
            m_interForces[3*index+k] += (pos[k]-pos2[k]) * u_p;
        }
    }
    r2 = 0;
    r2 += pos[0] * pos[0];
    r2 += pos[1] * pos[1];
    r2 += pos[2] * pos[2] * m_parameters[1] * m_parameters[1];
    m_distances[index][index] = r2;
    for (unsigned int j = index+1; j<particles.size(); j++){ // Column
        auto pos2 = particles[j]->getPosition();
        u_p = m_uPrime[j][index];
        for (unsigned int k = 0; k<3; k++)
            m_interForces[3*j+k] -= (pos2[k] - pos_old[k])*u_p;
        r2 = 0;
        for (unsigned int k = 0; k<3; k++){
            r2 += (pos2[k] - pos[k])*(pos2[k] - pos[k]);
        }
        r = sqrt(r2);
        m_distances[j][index]=r;
        u_p = uPrime_r(r);
        m_uPrime[j][index] = u_p;
        m_uDoublePrime[j][index] = uDoublePrime(r);
        for (unsigned int k = 0; k<3; k++){
            m_interForces[3*j+k] += (pos2[k] - pos[k])*u_p;
            m_interForces[3*index+k] += (pos[k]-pos2[k]) * u_p;
        }
    }
}

double InteractingGaussian3D::uPrime_r(double r){
    // *************** [Derivative of log(1.-(m_parameters[2]/r))]/r ***************
    double a = m_parameters[2];
    return a/(r*r*abs(a-r));
}

double InteractingGaussian3D::uDoublePrime(double r){
    // *************** Second derivative of log(1.-(m_parameters[2]/r)) ***************
    double a = m_parameters[2];
    return (a*(a-2*r))/(r*r*(a-r)*(a-r));
}

double InteractingGaussian3D::evaluate(std::vector<std::unique_ptr<class Particle>>& particles) {
    // Returns Phi, not Phi^2
    double r2 = 0, beta = m_parameters[1], a = m_parameters[2];
    for (unsigned int i = 0; i < particles.size(); i++){
        std::vector<double> position = particles[i]->getPosition();
        r2 += position[0]*position[0];
        r2 += position[1]*position[1];
        r2 += beta*position[2]*position[2];
    }
    double phi = exp(-1*r2*m_parameters[0]);
    for (unsigned int i = 0; i < particles.size(); i++){
        for (unsigned int j = i+1; j<particles.size(); j++){
            phi *= std::max(1-a/m_distances[j][i],0.);
        }
    }

    return phi;
}

double InteractingGaussian3D::computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles) {
    //***************WE RETURN d2/dx2(phi)/phi NOT d2/dx2(phi)*********************
    // The second derivative of exp(-alpha x*x) is exp(-alpha x*x)*(4*alpha*alpha*x*x - 2*alpha)
    
    // Non-interacting part
    double r2 = 0, alpha = m_parameters[0], beta = m_parameters[1];
    for (unsigned int i = 0; i < particles.size(); i++){
        r2 += m_distances[i][i];
    }
    int n = particles.size();
    double nabla2 = 4*alpha*alpha*r2 - (4+2*beta)*n*alpha;

    // double sum over all particles
    std::vector<double> repulsion, position;
    for (unsigned int k = 0; k < particles.size(); k++){
        for (unsigned int i = 0; i < k; i++){
            nabla2 += 2*m_uDoublePrime[k][i] + 4*m_uPrime[k][i];
        }
        position = particles[k]->getPosition();
        nabla2 += -4 * alpha * position[0] * m_interForces[3*k];
        nabla2 += m_interForces[3*k] * m_interForces[3*k];
        nabla2 += -4 * alpha * position[1] * m_interForces[3*k+1];
        nabla2 += m_interForces[3*k+1] * m_interForces[3*k+1];
        nabla2 += -4 * alpha * beta * position[2] * m_interForces[3*k+2];
        nabla2 += m_interForces[3*k+2] * m_interForces[3*k+2];
    }

    return nabla2;
}

std::vector<double> InteractingGaussian3D::quantumForce(std::vector<std::unique_ptr<class Particle>>& particles, int index){
    //***************WE RETURN d/dx(phi)/phi NOT d/dx(phi)*********************
    auto pos = particles[index]->getPosition();
    auto force = std::vector<double>(pos);
    double alpha = m_parameters[0], beta = m_parameters[1];
    force[0] *= -2*alpha;
    force[0] += m_interForces[3*index];
    force[1] *= -2*alpha;
    force[1] += m_interForces[3*index+1];
    force[2] *= -2*alpha*beta;
    force[2] += m_interForces[3*index+2];
    
    return force;
}

std::vector<double> InteractingGaussian3D::quantumForceMoved(std::vector<std::unique_ptr<class Particle>>& particles, int index, std::vector<double>& step){
    //***************WE RETURN d/dx(phi)/phi NOT d/dx(phi)*********************

    double r2, temp, alpha = m_parameters[0], beta = m_parameters[1];
    auto pos = particles[index]->getPosition();
    auto force = std::vector<double>(pos);
    force[0] += step[0];
    force[0] *= -2*alpha;
    force[1] += step[1];
    force[1] *= -2*alpha;
    force[2] += step[2];
    force[2] *= -2*alpha*beta;
    for (unsigned int i = 0; i<particles.size(); i++){
        if ((int)i == index) continue;
        auto pos2 = particles[i]->getPosition();
        auto relPos = std::vector<double>{0,0,0};
        r2 = 0;
        for (unsigned int k = 0; k<3; k++){
            relPos[k] = (pos[k] + step[k] - pos2[k]);
            r2 += relPos[k]*relPos[k];
        }
        temp = sqrt(r2);
        temp = uPrime_r(temp);
        for (unsigned int k = 0; k<3; k++){
            force[k] += relPos[k] * temp;
        }
    }
    return force;
}

double InteractingGaussian3D::phiRatio(std::vector<std::unique_ptr<class Particle>>& particles, int index, std::vector<double>& step){
    // Calculate (phi(new)/phi(old))**2
    double dr2 = 0, r2, r2new = 0, diff, Jik, JikNew, a = m_parameters[2], beta = m_parameters[1];
    auto pos = particles[index]->getPosition();
    dr2 += (2*pos[0]+step[0])*step[0]; 
    dr2 += (2*pos[1]+step[1])*step[1]; 
    dr2 += (2*pos[2]+step[2])*step[2]*beta; 
    double phi = exp(-2*dr2*m_parameters[0]);
    for (unsigned int i = 0; i<particles.size(); i++){
        if ((int)i == index) continue;
        auto pos2 = particles[i]->getPosition();
        r2 = 0;
        r2new = 0;
        for (unsigned int k = 0; k<3; k++){
            diff = pos[k] - pos2[k];
            r2 += diff*diff;
            r2new += (diff+step[k])*(diff+step[k]);
        }
        Jik = 1-a/sqrt(r2);
        if (Jik<0)
            phi *= 1E6; // arbitrary constant to motivate particles that overlap to move
        else{
            JikNew = std::max(1-a/sqrt(r2new),0.);
            phi *= JikNew*JikNew/(Jik*Jik);
        }
    }
    return phi;
}

std::vector<double> InteractingGaussian3D::getdPhi_dParams(std::vector<std::unique_ptr<class Particle>>& particles){
    double r2 = 0, z2 = 0, alpha = m_parameters[0], beta = m_parameters[1];
    for (unsigned int i = 0; i < particles.size(); i++){
        std::vector<double> position = particles[i]->getPosition();
        r2 += position[0]*position[0];
        r2 += position[1]*position[1];
        z2 += position[2]*position[2];
    }

    return std::vector<double>{-(r2+beta*z2), -alpha*z2};
}