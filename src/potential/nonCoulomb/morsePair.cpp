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

/**
 * @brief operator overload for the comparison of two MorsePair objects
 *
 * @param other
 * @return true
 * @return false
 */
bool MorsePair::operator==(const MorsePair &other) const
{
    return NonCoulombPair::operator==(other) &&
           utilities::compare(_dissociationEnergy, other._dissociationEnergy) &&
           utilities::compare(_wellWidth, other._wellWidth) &&
           utilities::compare(_equilibriumDistance, other._equilibriumDistance);
}

/**
 * @brief calculates the energy and force of a MorsePair
 *
 * @param distance
 * @return std::pair<double, double>
 */
std::pair<double, double> MorsePair::calculate(const double distance) const
{
    const auto expTerm =
        std::exp(-_wellWidth * (distance - _equilibriumDistance));

    const auto energy =
        _dissociationEnergy * (1.0 - expTerm) * (1.0 - expTerm) -
        _energyCutOff - _forceCutOff * (_radialCutOff - distance);
    const auto force =
        -2.0 * _dissociationEnergy * _wellWidth * expTerm * (1.0 - expTerm) -
        _forceCutOff;

    return {energy, force};
}