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

#include "nonCoulombPair.hpp"

#include "mathUtilities.hpp"   // for compare

using namespace potential;
using namespace utilities;

/**
 * @brief Construct a new Non Coulomb Pair:: Non Coulomb Pair object
 *
 * @param vanDerWaalsType1
 * @param vanDerWaalsType2
 * @param cutOff
 */
NonCoulombPair::NonCoulombPair(
    const size_t vanDerWaalsType1,
    const size_t vanDerWaalsType2,
    const double cutOff
)
    : _vanDerWaalsType1(vanDerWaalsType1),
      _vanDerWaalsType2(vanDerWaalsType2),
      _radialCutOff(cutOff)
{
}

/**
 * @brief Construct a new Non Coulomb Pair:: Non Coulomb Pair object
 *
 * @param cutOff
 */
NonCoulombPair::NonCoulombPair(const double cutOff) : _radialCutOff(cutOff) {}

/**
 * @brief Construct a new Non Coulomb Pair:: Non Coulomb Pair object
 *
 * @param cutoff
 * @param energyCutoff
 * @param forceCutoff
 */
NonCoulombPair::NonCoulombPair(
    const double cutoff,
    const double energyCutoff,
    const double forceCutoff
)
    : _radialCutOff(cutoff),
      _energyCutOff(energyCutoff),
      _forceCutOff(forceCutoff)
{
}

/**
 * @brief operator overload for the comparison of two NonCoulombPair objects
 *
 * @details returns also true if the two types are switched
 *
 * @note uses only the van der Waals types and the radial cut off
 *
 * @param other
 * @return true
 * @return false
 */
bool NonCoulombPair::operator==(const NonCoulombPair &other) const
{
    auto isEq = true;
    isEq      = isEq && _vanDerWaalsType1 == other._vanDerWaalsType1;
    isEq      = isEq && _vanDerWaalsType2 == other._vanDerWaalsType2;
    isEq      = isEq && compare(_radialCutOff, other._radialCutOff);

    auto isEqSymm = true;
    isEqSymm      = isEqSymm && _vanDerWaalsType1 == other._vanDerWaalsType2;
    isEqSymm      = isEqSymm && _vanDerWaalsType2 == other._vanDerWaalsType1;
    isEqSymm      = isEqSymm && compare(_radialCutOff, other._radialCutOff);

    return isEq || isEqSymm;
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set internal type 1
 *
 * @param internalType1
 */
void NonCoulombPair::setInternalType1(const size_t internalType1)
{
    _internalType1 = internalType1;
}

/**
 * @brief set internal type 2
 *
 * @param internalType2
 */
void NonCoulombPair::setInternalType2(const size_t internalType2)
{
    _internalType2 = internalType2;
}

/**
 * @brief set energy cut off
 *
 * @param energyCutoff
 */
void NonCoulombPair::setEnergyCutOff(const double energyCutoff)
{
    _energyCutOff = energyCutoff;
}

/**
 * @brief set force cut off
 *
 * @param forceCutoff
 */
void NonCoulombPair::setForceCutOff(const double forceCutoff)
{
    _forceCutOff = forceCutoff;
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief get van der Waals type 1
 *
 * @return size_t
 */
size_t NonCoulombPair::getVanDerWaalsType1() const { return _vanDerWaalsType1; }

/**
 * @brief get van der Waals type 2
 *
 * @return size_t
 */
size_t NonCoulombPair::getVanDerWaalsType2() const { return _vanDerWaalsType2; }

/**
 * @brief get internal type 1
 *
 * @return size_t
 */
size_t NonCoulombPair::getInternalType1() const { return _internalType1; }

/**
 * @brief get internal type 2
 *
 * @return size_t
 */
size_t NonCoulombPair::getInternalType2() const { return _internalType2; }

/**
 * @brief get radial cut off
 *
 * @return double
 */
double NonCoulombPair::getRadialCutOff() const { return _radialCutOff; }

/**
 * @brief get energy cut off
 *
 * @return double
 */
double NonCoulombPair::getEnergyCutOff() const { return _energyCutOff; }

/**
 * @brief get force cut off
 *
 * @return double
 */
double NonCoulombPair::getForceCutOff() const { return _forceCutOff; }