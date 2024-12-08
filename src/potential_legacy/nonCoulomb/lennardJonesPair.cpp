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

#include "lennardJonesPair.hpp"

#include "mathUtilities.hpp"   // for compare

using namespace potential;
using namespace utilities;

/**
 * @brief Construct a new Lennard Jones Pair:: Lennard Jones Pair object
 *
 * @param vanDerWaalsType1
 * @param vanDerWaalsType2
 * @param cutOff
 * @param c6
 * @param c12
 */
LennardJonesPair::LennardJonesPair(
    const size_t vanDerWaalsType1,
    const size_t vanDerWaalsType2,
    const double cutOff,
    const double c6,
    const double c12
)
    : NonCoulombPair(vanDerWaalsType1, vanDerWaalsType2, cutOff),
      _c6(c6),
      _c12(c12) {};

/**
 * @brief Construct a new Lennard Jones Pair:: Lennard Jones Pair object
 *
 * @param cutOff
 * @param c6
 * @param c12
 */
LennardJonesPair::LennardJonesPair(
    const double cutOff,
    const double c6,
    const double c12
)
    : NonCoulombPair(cutOff), _c6(c6), _c12(c12) {};

/**
 * @brief Construct a new Lennard Jones Pair:: Lennard Jones Pair object
 *
 * @param cutOff
 * @param energyCutoff
 * @param forceCutoff
 * @param c6
 * @param c12
 */
LennardJonesPair::LennardJonesPair(
    const double cutOff,
    const double energyCutoff,
    const double forceCutoff,
    const double c6,
    const double c12
)
    : NonCoulombPair(cutOff, energyCutoff, forceCutoff), _c6(c6), _c12(c12) {};

/**
 * @brief operator overload for the comparison of two LennardJonesPair objects
 *
 * @param other
 * @return true
 * @return false
 */
bool LennardJonesPair::operator==(const LennardJonesPair &other) const
{
    auto isEqual = true;
    isEqual      = isEqual && NonCoulombPair::operator==(other);
    isEqual      = isEqual && compare(_c6, other._c6);
    isEqual      = isEqual && compare(_c12, other._c12);

    return isEqual;
}

/**
 * @brief calculates the energy and force of a LennardJonesPair
 *
 * @param distance
 * @return std::pair<double, double>
 */
std::pair<double, double> LennardJonesPair::calculate(const double distance
) const
{
    const auto distanceThird   = distance * distance * distance;
    const auto distanceSixth   = distanceThird * distanceThird;
    const auto distanceTwelfth = distanceSixth * distanceSixth;

    auto energy  = _c12 / distanceTwelfth;
    energy      += _c6 / distanceSixth;
    energy      -= _energyCutOff;
    energy      -= _forceCutOff * (_radialCutOff - distance);

    auto force  = 12.0 * _c12 / (distanceTwelfth * distance);
    force      += 6.0 * _c6 / (distanceSixth * distance);
    force      -= _forceCutOff;

    return {energy, force};
}

/**
 * @brief get the c6 and c12 coefficients
 *
 * @return double
 */
double LennardJonesPair::getC6() const { return _c6; }

/**
 * @brief get the c6 and c12 coefficients
 *
 * @return double
 */
double LennardJonesPair::getC12() const { return _c12; }