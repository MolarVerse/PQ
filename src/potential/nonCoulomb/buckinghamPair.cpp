/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "buckinghamPair.hpp"

#include "mathUtilities.hpp"   // for compare

#include <cmath>   // for exp

using namespace potential;

/**
 * @brief operator overload for the comparison of two BuckinghamPair objects
 *
 * @param other
 * @return true
 * @return false
 */
bool BuckinghamPair::operator==(const BuckinghamPair &other) const
{
    return NonCoulombPair::operator==(other) && utilities::compare(_a, other._a) && utilities::compare(_dRho, other._dRho) &&
           utilities::compare(_c6, other._c6);
}

/**
 * @brief calculates the energy and force of a BuckinghamPair
 *
 * @link https://doi.org/10.1098/rspa.1938.0173
 *
 * @param distance
 * @return std::pair<double, double>
 */
std::pair<double, double> BuckinghamPair::calculateEnergyAndForce(const double distance) const
{
    const auto distanceSixth = distance * distance * distance * distance * distance * distance;
    const auto expTerm       = _a * std::exp(_dRho * distance);

    const auto energy = expTerm + _c6 / distanceSixth - _energyCutOff - _forceCutOff * (_radialCutOff - distance);
    const auto force  = -_dRho * expTerm + 6.0 * _c6 / (distanceSixth * distance) - _forceCutOff;

    return {energy, force};
}