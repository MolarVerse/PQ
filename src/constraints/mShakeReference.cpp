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

#include "mShakeReference.hpp"

#include <memory>   // for make_shared

#include "atom.hpp"            // for Atom
#include "exceptions.hpp"      // for MShakeFileException
#include "moleculeType.hpp"    // for MoleculeType
#include "simulationBox.hpp"   // for SimulationBox

using namespace constraints;
using namespace simulationBox;

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief Set the molecule type object as a unique pointer
 *
 * @param moltype
 */
void MShakeReference::setMoleculeType(MoleculeType &moltype)
{
    _moleculeType = std::make_shared<MoleculeType>(moltype);
}

/**
 * @brief Set the atoms object
 *
 * @param atoms
 */
void MShakeReference::setAtoms(const std::vector<Atom> &atoms)
{
    _atoms = atoms;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the number of atoms
 *
 * @return const size_t
 */
size_t MShakeReference::getNumberOfAtoms() const { return _atoms.size(); }

/**
 * @brief get the atoms
 *
 * @return const std::vector<Atom>&
 */
std::vector<Atom> &MShakeReference::getAtoms() { return _atoms; }

/**
 * @brief get the molecule type
 *
 * @return const MoleculeType&
 */
MoleculeType &MShakeReference::getMoleculeType() const
{
    return *_moleculeType;
}