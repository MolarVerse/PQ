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
    auto isEqual = _vanDerWaalsType1 == other._vanDerWaalsType1;
    isEqual      = isEqual && _vanDerWaalsType2 == other._vanDerWaalsType2;
    isEqual      = isEqual && utilities::compare(_radialCutOff, other._radialCutOff);

    auto isEqualSymmetric = _vanDerWaalsType1 == other._vanDerWaalsType2;
    isEqualSymmetric      = isEqualSymmetric && _vanDerWaalsType2 == other._vanDerWaalsType1;
    isEqualSymmetric      = isEqualSymmetric && utilities::compare(_radialCutOff, other._radialCutOff);

    return isEqual || isEqualSymmetric;
}