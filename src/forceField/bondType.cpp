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

#include "bondType.hpp"

#include "mathUtilities.hpp"

using namespace forceField;
using namespace utilities;

/**
 * @brief Construct a new Bond Type:: Bond Type object
 *
 * @param id
 * @param equilibriumBondLength
 * @param springConstant
 */
BondType::BondType(
    const size_t id,
    const double equilibriumBondLength,
    const double springConstant
)
    : _id(id),
      _equilBondLength(equilibriumBondLength),
      _forceConstant(springConstant)
{
}

/**
 * @brief operator overload for the comparison of two BondType objects
 *
 * @param other
 * @return true
 * @return false
 */
bool forceField::operator==(const BondType &self, const BondType &other)
{
    auto isEq = self._id == other._id;
    isEq      = isEq && compare(self._equilBondLength, other._equilBondLength);
    isEq      = isEq && compare(self._forceConstant, other._forceConstant);

    return isEq;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the id of the bond type
 *
 * @return size_t
 */
size_t BondType::getId() const { return _id; }

/**
 * @brief get the equilibrium bond length of the bond type
 *
 * @return double
 */
double BondType::getEquilibriumBondLength() const { return _equilBondLength; }

/**
 * @brief get the force constant of the bond type
 *
 * @return double
 */
double BondType::getForceConstant() const { return _forceConstant; }