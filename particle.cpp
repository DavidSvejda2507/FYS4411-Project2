#include "particle.h"
#include <assert.h>

Particle::Particle(const std::vector<double> &position)
{
    m_numberOfDimensions = position.size();
    m_position = position;
}

void Particle::adjustPosition(double change, unsigned int dimension)
{
    m_position.at(dimension) += change;
}

void Particle::adjustPosition(std::vector<double> step)
{
    for (unsigned int i = 0; i < m_position.size(); i++)
        m_position[i] += step[i];
}
