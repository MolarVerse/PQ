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

using namespace forceField;

/**
 * @brief Construct a new Bond Type:: Bond Type object
 *
 * @param id
 * @param params
 */
BondType::BondType(const BondId id, const BondParams &params)
    : _id(id), _params(params)
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
    return self._id == other._id && self._params == other._params;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the id of the bond type
 *
 * @return BondId
 */
BondId BondType::getId() const { return _id; }

/**
 * @brief get the parameters of the bond type
 *
 * @return const BondParams&
 */
const BondParams &BondType::getParams() const { return _params; }
