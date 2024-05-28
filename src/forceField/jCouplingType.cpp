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

#include "jCouplingType.hpp"

#include "mathUtilities.hpp"

using namespace forceField;

/**
 * @brief operator overload for the comparison of two JCouplingType objects
 *
 * @param other
 * @return true
 * @return false
 */
bool JCouplingType::operator==(const JCouplingType &other) const
{
    const auto k         = _forceConstant;
    const auto other_k   = other._forceConstant;
    const auto phi       = _phaseShift;
    const auto other_phi = other._phaseShift;

    auto isEqual = _id == other._id;
    isEqual      = isEqual && utilities::compare(_J0, other._J0);
    isEqual      = isEqual && utilities::compare(k, other_k);
    isEqual      = isEqual && utilities::compare(_a, other._a);
    isEqual      = isEqual && utilities::compare(_b, other._b);
    isEqual      = isEqual && utilities::compare(_c, other._c);
    isEqual      = isEqual && utilities::compare(phi, other_phi);

    return isEqual;
}