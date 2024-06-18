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

#include "bond.hpp"

using namespace connectivity;
using namespace simulationBox;

/**
 * @brief Construct a new Bond:: Bond object
 *
 * @param molecule1
 * @param molecule2
 * @param atomIndex1
 * @param atomIndex2
 */
Bond::Bond(
    Molecule    *molecule1,
    Molecule    *molecule2,
    const size_t atomIndex1,
    const size_t atomIndex2
)
    : ConnectivityElement({molecule1, molecule2}, {atomIndex1, atomIndex2})
{
}

/**
 * @brief Construct a new Bond:: Bond object
 *
 * @param molecule1
 * @param atomIndex1
 * @param molecule2
 * @param atomIndex2
 */
Bond::Bond(
    Molecule    *molecule1,
    const size_t atomIndex1,
    Molecule    *molecule2,
    const size_t atomIndex2
)
    : ConnectivityElement({molecule1, molecule2}, {atomIndex1, atomIndex2})
{
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief Get the molecule1 object
 *
 * @return Molecule*
 */
Molecule *Bond::getMolecule1() const { return _molecules[0]; }

/**
 * @brief Get the molecule2 object
 *
 * @return Molecule*
 */
Molecule *Bond::getMolecule2() const { return _molecules[1]; }

/**
 * @brief Get the atomIndex1 object
 *
 * @return size_t
 */
size_t Bond::getAtomIndex1() const { return _atomIndices[0]; }

/**
 * @brief Get the atomIndex2 object
 *
 * @return size_t
 */
size_t Bond::getAtomIndex2() const { return _atomIndices[1]; }