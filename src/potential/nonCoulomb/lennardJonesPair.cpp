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

/**
 * @brief operator overload for the comparison of two LennardJonesPair objects
 *
 * @param other
 * @return true
 * @return false
 */
bool LennardJonesPair::operator==(const LennardJonesPair &other) const
{
    return NonCoulombPair::operator==(other) &&
           utilities::compare(_c6, other._c6) &&
           utilities::compare(_c12, other._c12);
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
    const auto distanceSquared = distance * distance;
    const auto distanceSixth =
        distanceSquared * distanceSquared * distanceSquared;
    const auto distanceTwelfth = distanceSixth * distanceSixth;

    const auto energy = _c12 / distanceTwelfth + _c6 / distanceSixth -
                        _energyCutOff -
                        _forceCutOff * (_radialCutOff - distance);
    const auto force = 12.0 * _c12 / (distanceTwelfth * distance) +
                       6.0 * _c6 / (distanceSixth * distance) - _forceCutOff;

    return {energy, force};
}