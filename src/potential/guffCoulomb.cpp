#include "coulombPotential.hpp"

#include <cmath>
#include <iostream>

using namespace potential;

/**
 * @brief calculates the coulomb potential and its force for the guff-coulomb potential
 *
 * @param coulombCoefficient
 * @param rcCutoff
 * @param distance
 * @param energy
 * @param force
 * @param energyCutoff
 * @param forceCutoff
 */
void GuffCoulomb::calcCoulomb(const double coulombCoefficient,
                              const double distance,
                              double      &energy,
                              double      &force,
                              const double energyCutoff,
                              const double forceCutoff) const
{
    energy  = coulombCoefficient * (1 / distance) - energyCutoff - forceCutoff * (_coulombRadiusCutOff - distance);
    force  += coulombCoefficient * (1 / (distance * distance)) - forceCutoff;
}

/**
 * @brief Constructor for GuffWolfCoulomb
 *
 * @details calculates Wolf parameters
 *
 * @param coulombRadiusCutOff
 * @param kappa
 */
GuffWolfCoulomb::GuffWolfCoulomb(const double coulombRadiusCutOff, const double kappa)
    : CoulombPotential::CoulombPotential(coulombRadiusCutOff), _kappa(kappa)
{
    _wolfParameter1 = ::erfc(_kappa * coulombRadiusCutOff) / coulombRadiusCutOff;
    _wolfParameter2 = 2.0 * _kappa / ::sqrt(M_PI);
    _wolfParameter3 = _wolfParameter1 / coulombRadiusCutOff +
                      _wolfParameter2 * ::exp(-_kappa * _kappa * coulombRadiusCutOff * coulombRadiusCutOff) / coulombRadiusCutOff;
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