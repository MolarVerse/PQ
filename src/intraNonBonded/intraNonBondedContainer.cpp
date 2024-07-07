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

#include "intraNonBondedContainer.hpp"

using namespace intraNonBonded;

/**
 * @brief constructor for IntraNonBondedContainer
 *
 * @param molType
 * @param atomIndices
 */
IntraNonBondedContainer::IntraNonBondedContainer(
    const size_t                         molType,
    const std::vector<std::vector<int>> &atomIndices
)
    : _molType(molType), _atomIndices(atomIndices){};

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the molType
 *
 * @return size_t
 */
size_t IntraNonBondedContainer::getMolType() const { return _molType; }

/**
 * @brief get the atomIndices
 *
 * @return std::vector<std::vector<int>>
 */
std::vector<std::vector<int>> IntraNonBondedContainer::getAtomIndices() const
{
    return _atomIndices;
}