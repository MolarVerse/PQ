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

#include "dihedralType.hpp"

#include "mathUtilities.hpp"

using namespace forceField;
using namespace utilities;

/**
 * @brief Construct a new Dihedral Type:: Dihedral Type object
 *
 * @param id
 * @param forceConstant
 * @param frequency
 * @param phaseShift
 */
DihedralType::DihedralType(
    const size_t id,
    const double forceConstant,
    const double frequency,
    const double phaseShift
)
    : _id(id),
      _forceConstant(forceConstant),
      _periodicity(frequency),
      _phaseShift(phaseShift)
{
}

/**
 * @brief operator overload for the comparison of two DihedralType objects
 *
 * @param other
 * @return true
 * @return false
 */
bool forceField::operator==(const DihedralType &self, const DihedralType &other)
{
    auto isEqual = self._id == other._id;
    isEqual = isEqual && compare(self._forceConstant, other._forceConstant);
    isEqual = isEqual && compare(self._periodicity, other._periodicity);
    isEqual = isEqual && compare(self._phaseShift, other._phaseShift);

    return isEqual;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the id of the dihedral type
 *
 * @return size_t
 */
size_t DihedralType::getId() const { return _id; }

/**
 * @brief get the force constant of the dihedral type
 *
 * @return double
 */
double DihedralType::getForceConstant() const { return _forceConstant; }

/**
 * @brief get the periodicity of the dihedral type
 *
 * @return double
 */
double DihedralType::getPeriodicity() const { return _periodicity; }

/**
 * @brief get the phase shift of the dihedral type
 *
 * @return double
 */
double DihedralType::getPhaseShift() const { return _phaseShift; }