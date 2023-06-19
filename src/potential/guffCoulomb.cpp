#include "coulombPotential.hpp"

using namespace potential;

void GuffCoulomb::calcCoulomb(const double coulombCoefficient,
                              const double rcCutoff,
                              const double distance,
                              double      &energy,
                              double      &force,
                              const double energy_cutoff,
                              const double force_cutoff) const
{
    energy = coulombCoefficient * (1 / distance) - energy_cutoff - force_cutoff * (rcCutoff - distance);
    force += coulombCoefficient * (1 / (distance * distance)) - force_cutoff;
}