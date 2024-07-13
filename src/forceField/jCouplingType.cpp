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
 * @brief Construct a new JCouplingType::JCouplingType object
 *
 * @param id
 * @param J0
 * @param forceConstant
 * @param a
 * @param b
 * @param c
 */
JCouplingType::JCouplingType(
    const size_t id,
    const double J0,
    const double forceConstant,
    const double a,
    const double b,
    const double c
)
    : _id(id), _J0(J0), _forceConstant(forceConstant), _a(a), _b(b), _c(c)
{
}

/**
 * @brief operator overload for the comparison of two JCouplingType objects
 *
 * @param self
 * @param other
 * @return true
 * @return false
 */
bool forceField::operator==(
    const JCouplingType &self,
    const JCouplingType &other
)
{
    const auto k       = self._forceConstant;
    const auto other_k = other._forceConstant;

    auto isEqual = self._id == other._id;
    isEqual      = isEqual && utilities::compare(self._J0, other._J0);
    isEqual      = isEqual && utilities::compare(k, other_k);
    isEqual      = isEqual && utilities::compare(self._a, other._a);
    isEqual      = isEqual && utilities::compare(self._b, other._b);
    isEqual      = isEqual && utilities::compare(self._c, other._c);

    return isEqual;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the id
 *
 * @return size_t
 */
size_t JCouplingType::getId() const { return _id; }

/**
 * @brief get the J0
 *
 * @return double
 */
double JCouplingType::getJ0() const { return _J0; }

/**
 * @brief get the force constant
 *
 * @return double
 */
double JCouplingType::getForceConstant() const { return _forceConstant; }

/**
 * @brief get the a
 *
 * @return double
 */
double JCouplingType::getA() const { return _a; }

/**
 * @brief get the b
 *
 * @return double
 */
double JCouplingType::getB() const { return _b; }

/**
 * @brief get the c
 *
 * @return double
 */
double JCouplingType::getC() const { return _c; }

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set if the upper symmetry should be used
 *
 * @param boolean
 */
void JCouplingType::setUpperSymmetry(const bool boolean)
{
    _upperSymmetry = boolean;
}

/**
 * @brief set if the lower symmetry should be used
 *
 * @param boolean
 */
void JCouplingType::setLowerSymmetry(const bool boolean)
{
    _lowerSymmetry = boolean;
}