#include "coulombPotential.hpp"

#include <cmath>

using namespace potential;

void GuffCoulomb::calcCoulomb(const double coulombCoefficient,
                              const double distance,
                              double      &energy,
                              double      &force,
                              const double energy_cutoff,
                              const double force_cutoff) const
{
    energy  = coulombCoefficient * (1 / distance) - energy_cutoff - force_cutoff * (_coulombRadiusCutOff - distance);
    force  += coulombCoefficient * (1 / (distance * distance)) - force_cutoff;
}

/**
 * @brief Constructor for GuffWolfCoulomb
 *
 * @details calculates Wolf parameters
 *
 * @param coulombRadiusCutoff
 * @param kappa
 */
GuffWolfCoulomb::GuffWolfCoulomb(const double coulombRadiusCutoff, const double kappa)
    : CoulombPotential::CoulombPotential(coulombRadiusCutoff), _kappa(kappa)
{
    _wolfParameter1 = ::erfc(_kappa * coulombRadiusCutoff) / coulombRadiusCutoff;
    _wolfParameter2 = 2.0 * _kappa / ::sqrt(M_PI);
    _wolfParameter3 = _wolfParameter1 / coulombRadiusCutoff +
                      _wolfParameter2 * ::exp(-_kappa * _kappa * coulombRadiusCutoff * coulombRadiusCutoff) / coulombRadiusCutoff;
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
void GuffWolfCoulomb::calcCoulomb(
    const double coulombCoefficient, const double distance, double &energy, double &force, const double, const double) const
{
    const auto kappaDistance = _kappa * distance;
    const auto erfcFactor    = ::erfc(kappaDistance);

    energy = coulombCoefficient * (erfcFactor / distance - _wolfParameter1 + _wolfParameter3 * (distance - _coulombRadiusCutOff));
    force += coulombCoefficient * (erfcFactor / (distance * distance) +
                                   _wolfParameter2 * ::exp(-kappaDistance * kappaDistance) / distance - _wolfParameter3);
}