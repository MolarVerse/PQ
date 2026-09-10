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

using namespace forceField;

/**
 * @brief Construct a new JCouplingType::JCouplingType object
 *
 * @param id
 * @param params
 */
JCouplingType::JCouplingType(const size_t id, const JCouplingParams &params)
    : _id(id), _params(params)
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
    return self._id == other._id && self._params == other._params;
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
 * @brief get the JCouplingParams
 *
 * @return const JCouplingParams&
 */
const JCouplingParams &JCouplingType::getParams() const { return _params; }

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
