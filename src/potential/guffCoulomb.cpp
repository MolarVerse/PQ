#include "coulombPotential.hpp"

#include <cmath>

using namespace potential;

void GuffCoulomb::calcCoulomb(const double coulombCoefficient,
                              const double rcCutoff,
                              const double distance,
                              double      &energy,
                              double      &force,
                              const double energy_cutoff,
                              const double force_cutoff) const
{
    energy  = coulombCoefficient * (1 / distance) - energy_cutoff - force_cutoff * (rcCutoff - distance);
    force  += coulombCoefficient * (1 / (distance * distance)) - force_cutoff;
}

/**
 * @brief calculates coulomb potential with Wolf long range correction
 *
 * @details
 *  TODO: move wolfParameters to initialization list
 *
 * @param coulombCoefficient
 * @param rcCutoff
 * @param distance
 * @param energy
 * @param force
 * @param dummy_energy_cutoff
 * @param dummy_force_cutoff
 */
void GuffWolfCoulomb::calcCoulomb(const double coulombCoefficient,
                                  const double rcCutoff,
                                  const double distance,
                                  double      &energy,
                                  double      &force,
                                  const double,
                                  const double) const
{
    const auto kappaDistance  = _kappa * distance;
    const auto erfcFactor     = ::erfc(kappaDistance);
    const auto wolfParameter1 = ::erfc(_kappa * rcCutoff) / rcCutoff;
    const auto wolfParameter2 = 2.0 * _kappa / ::sqrt(M_PI);
    const auto wolfParameter3 =
        wolfParameter1 / rcCutoff + wolfParameter2 * ::exp(-_kappa * _kappa * rcCutoff * rcCutoff) / rcCutoff;

    energy  = coulombCoefficient * (erfcFactor / distance - wolfParameter1 + wolfParameter3 * (distance - rcCutoff));
    force  += coulombCoefficient * (erfcFactor / (distance * distance) +
                                   wolfParameter2 * ::exp(-kappaDistance * kappaDistance) / distance - wolfParameter3);
}