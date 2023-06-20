#pragma once
#include "sampler.h"
#include <fstream>
class SamplerFineTune2D : public Sampler
{
public:
    // this sampler is a simplified version of Sampler which does not sample the quantities used to compute the gradient.
    SamplerFineTune2D(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        int numberOfWFParams,
        double scale);
    SamplerFineTune2D(std::vector<std::unique_ptr<class SamplerFineTune2D>> &samplers);
    ~SamplerFineTune2D();

    void writeHistogram();
    void sample(bool acceptedStep, System *system) override;

private:
    std::ofstream m_outBinaryFile;
    unsigned long int **m_position_histogram;
    int m_nx = 200;
    int m_ny = 200;
    double m_Min, m_Max;
};
