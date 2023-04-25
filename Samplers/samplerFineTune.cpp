#include <memory>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <omp.h>
#include "system.h"
#include "sampler.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"
#include "samplerFineTune.h"
#include "utils.h"
using std::cout;
using std::endl;


SamplerFineTune::SamplerFineTune( unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    int ) : Sampler::Sampler(numberOfParticles, numberOfDimensions, 0)
{
    int thread_number = omp_get_thread_num();
    std::string fname_thread = "./Outputs/sampledEnergies_" + std::to_string(numberOfParticles) + "_" + std::to_string(thread_number) + ".bin";
    m_outBinaryFile.open(fname_thread, std::ios::binary);
    m_position_histogram = init_3d_array(m_nx, m_ny, m_nz);
    m_xMin = -1.5;
    m_xMax = 1.5;
    m_yMin = -1.5;
    m_yMax = +1.5;
    m_zMin = -1.5;
    m_zMax = +1.5;
}

SamplerFineTune::SamplerFineTune(std::vector<std::unique_ptr< class SamplerFineTune >  >  & samplers ) :
     Sampler()
{

    m_position_histogram = init_3d_array(m_nx, m_ny, m_nz);
    int numberOfWFParams = samplers[0]->getGradientTerms().size();
    m_numberOfDimensions = samplers[0]->getNdim();
    m_waveFunctionParameters = samplers[0]->getWFparams();
    m_numberOfParticles = samplers[0]->getNparticles();

    std::string fname_histogram = "./Outputs/PositionHistogram_" + std::to_string(m_numberOfParticles) + ".bin";
    m_outBinaryFile.open(fname_histogram, std::ios::binary);

    m_gradientTerms = std::vector<std::vector<double>>(numberOfWFParams, std::vector<double>(2, 0.0)) ;
    int Nparams = m_gradientTerms.size();
    int Nsamplers = samplers.size();
    for(auto& sampler : samplers){
        //sum sampled energy
        m_energy += sampler->getEnergy();
        m_energy2+= sampler->getEnergy2();
        //sum accepted nr. of steps
        m_stepNumber += sampler->getNSteps();
        m_numberOfAcceptedSteps+= sampler->getNAccSteps();
        m_equilibrationStepNumber+=  sampler->getNStepsEq();
        m_numberOfAcceptedEquilibrationSteps+= sampler->getNAccStepsEq();
        //sum sampled gradients
        std::vector<std::vector<double>> cur_gradient = sampler->getGradientTerms();
        for(int j=0; j<Nparams; j++){
            for(int k=0; k<2; k++){
                m_gradientTerms[j][k]+=cur_gradient[j][k];
            }
        }
        //update histogram of positions. 
        for(int i=0; i<m_nx; i++){
            for(int j=0; j<m_ny; j++){
                for(int k=0; k<m_nz; k++){
                    m_position_histogram[i][j][k]+=sampler->m_position_histogram[i][j][k];
                }
            }
        }
        //obtain averages
        m_energy/=Nsamplers;
        m_energy2/=Nsamplers;
        for(int j=0; j<Nparams; j++){
            for(int k=0; k<2; k++){
                m_gradientTerms[j][k]/=Nsamplers;
            }
        }
    }
}
void SamplerFineTune::sample(bool acceptedStep, System* system) {
    /*sample all the interesting things 
     */
    double localEnergy = system->computeLocalEnergy();
    m_cumulativeEnergy  += localEnergy;
    m_cumulativeEnergy2 += localEnergy * localEnergy;

    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;
    std::vector<double> pos = system->getParticlePosition(0);
    int i,j,k;

    i = (int) ((pos[0]-m_xMin)/(m_xMax-m_xMin)*m_nx);
    j = (int) ((pos[1]-m_yMin)/(m_yMax-m_yMin)*m_ny);
    k = (int) ((pos[2]-m_zMin)/(m_zMax-m_zMin)*m_nz);
    bool within_limits = (i<m_nx) & (j<m_ny) & (k<m_nz) & (i>0) & (j>0) & (k>0);
    if (within_limits)
        m_position_histogram[i][j][k]+=1;
    //write sampled energy to a file
    m_outBinaryFile.write(reinterpret_cast<const char*>(&localEnergy), sizeof(double));

}

void SamplerFineTune::writeHistogram(){
    int thread_number = omp_get_thread_num();
    std::string fname_thread = "./Outputs/histogram" + std::to_string(m_numberOfParticles) + "_" + std::to_string(thread_number) + ".bin";
    //file may be already open to write the energies.
    if (m_outBinaryFile.is_open()) {
        m_outBinaryFile.close();
    }
    m_outBinaryFile.open(fname_thread , std::ios::binary );
    unsigned int hist_size = m_nx*m_ny*m_nz*sizeof(unsigned int);
    m_outBinaryFile.write((char*) m_position_histogram[0][0], hist_size);
    return ; 
}
