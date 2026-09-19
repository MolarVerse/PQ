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

#include "jCouplingForceField.hpp"

#include "strongTypes.hpp"

using namespace forceField;
using namespace connectivity;

/**
 * @brief Construct a new JCouplingForceField::JCouplingForceField object
 *
 * @param molecules
 * @param atomIndices
 * @param type
 */
JCouplingForceField::JCouplingForceField(
    const std::vector<molsys::Molecule *> &molecules,
    const std::vector<AtomIndex>          &atomIndices,
    size_t                                 type
)
    : Dihedral(molecules, atomIndices), _type(type), _params(std::nullopt)
{
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief Set the upper symmetry
 *
 * @param boolean
 */
void JCouplingForceField::setUpperSymmetry(bool boolean)
{
    _upperSymmetry = boolean;
}

/**
 * @brief Set the lower symmetry
 *
 * @param boolean
 */
void JCouplingForceField::setLowerSymmetry(bool boolean)
{
    _lowerSymmetry = boolean;
}

/**
 * @brief Set the JCoupling parameters
 *
 * @param params
 */
void JCouplingForceField::setParams(const JCouplingParams &params)
{
    _params = params;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the type
 *
 * @return size_t
 */
size_t JCouplingForceField::getType() const { return _type; }

/**
 * @brief get if the upper symmetry is set
 *
 * @return bool
 */
bool JCouplingForceField::getUpperSymmetry() const { return _upperSymmetry; }

/**
 * @brief get if the lower symmetry is set
 *
 * @return bool
 */
bool JCouplingForceField::getLowerSymmetry() const { return _lowerSymmetry; }
