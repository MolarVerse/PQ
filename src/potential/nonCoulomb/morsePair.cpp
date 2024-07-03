/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "morsePair.hpp"

#include <cmath>   // for exp

#include "mathUtilities.hpp"   // for compare

using namespace potential;
using namespace utilities;

/**
 * @brief Construct a new Morse Pair:: Morse Pair object
 *
 * @param vanDerWaalsType1
 * @param vanDerWaalsType2
 * @param cutOff
 * @param dissociationEnergy
 * @param wellWidth
 * @param equilibriumDistance
 */
MorsePair::MorsePair(
    const size_t vanDerWaalsType1,
    const size_t vanDerWaalsType2,
    const double cutOff,
    const double dissociationEnergy,
    const double wellWidth,
    const double equilibriumDistance
)
    : NonCoulombPair(vanDerWaalsType1, vanDerWaalsType2, cutOff),
      _dissociationEnergy(dissociationEnergy),
      _wellWidth(wellWidth),
      _equilibriumDistance(equilibriumDistance){};

/**
 * @brief Construct a new Morse Pair:: Morse Pair object
 *
 * @param cutOff
 * @param dissociationEnergy
 * @param wellWidth
 * @param equilibriumDistance
 */
MorsePair::MorsePair(
    const double cutOff,
    const double dissociationEnergy,
    const double wellWidth,
    const double equilibriumDistance
)
    : NonCoulombPair(cutOff),
      _dissociationEnergy(dissociationEnergy),
      _wellWidth(wellWidth),
      _equilibriumDistance(equilibriumDistance){};

/**
 * @brief Construct a new Morse Pair:: Morse Pair object
 *
 * @param cutOff
 * @param energyCutoff
 * @param forceCutoff
 * @param dissociationEnergy
 * @param wellWidth
 * @param equilibriumDistance
 */
MorsePair::MorsePair(
    const double cutOff,
    const double energyCutoff,
    const double forceCutoff,
    const double dissociationEnergy,
    const double wellWidth,
    const double equilibriumDistance
)
    : NonCoulombPair(cutOff, energyCutoff, forceCutoff),
      _dissociationEnergy(dissociationEnergy),
      _wellWidth(wellWidth),
      _equilibriumDistance(equilibriumDistance){};

/**
 * @brief operator overload for the comparison of two MorsePair objects
 *
 * @param other
 * @return true
 * @return false
 */
bool MorsePair::operator==(const MorsePair &other) const
{
    auto isEq = true;

    isEq = isEq && NonCoulombPair::operator==(other);
    isEq = isEq && compare(_dissociationEnergy, other._dissociationEnergy);
    isEq = isEq && compare(_wellWidth, other._wellWidth);
    isEq = isEq && compare(_equilibriumDistance, other._equilibriumDistance);

    return isEq;
}

/**
 * @brief calculates the energy and force of a MorsePair
 *
 * @param distance
 * @return std::pair<double, double>
 */
std::pair<double, double> MorsePair::calculate(const double distance) const
{
    const auto deltaEquilibrium = distance - _equilibriumDistance;
    const auto expTerm          = std::exp(-_wellWidth * deltaEquilibrium);
    const auto oneMinusExpTerm  = 1.0 - expTerm;

    auto energy  = _dissociationEnergy * oneMinusExpTerm * oneMinusExpTerm;
    energy      -= _energyCutOff;
    energy      -= _forceCutOff * (_radialCutOff - distance);

    auto force  = -2.0 * _dissociationEnergy * _wellWidth;
    force      *= expTerm * oneMinusExpTerm;
    force      -= _forceCutOff;

    return {energy, force};
}

/**
 * @brief get the dissociation energy
 *
 * @return double
 */
double MorsePair::getDissociationEnergy() const { return _dissociationEnergy; }

/**
 * @brief get the well width
 *
 * @return double
 */
double MorsePair::getWellWidth() const { return _wellWidth; }

/**
 * @brief get the equilibrium distance
 *
 * @return double
 */
double MorsePair::getEquilibriumDistance() const
{
    return _equilibriumDistance;
}