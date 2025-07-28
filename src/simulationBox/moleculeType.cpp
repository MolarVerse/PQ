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

#include "moleculeType.hpp"

#include <algorithm>   // for sort, unique
#include <iterator>    // for std::ranges::size
#include <ranges>      // for ranges::size, ranges::unique

using namespace simulationBox;

/**
 * @brief Construct a new Molecule Type:: Molecule Type object
 *
 * @param moltype
 */
MoleculeType::MoleculeType(const size_t moltype) : _moltype(moltype) {}

/**
 * @brief Construct a new Molecule Type:: Molecule Type object
 *
 * @param name
 */
MoleculeType::MoleculeType(const std::string_view &name) : _name(name) {}

/**
 * @brief finds number of different atom types in molecule
 *
 * @note This function cannot be const due to std::ranges::unique
 *
 * @return int
 */
size_t MoleculeType::MoleculeType::getNumberOfAtomTypes()
{
    const auto nUnique = std::ranges::size(std::ranges::unique(_atomTypes));

    return _externalAtomTypes.size() - nUnique;
}

/**************************
 *                        *
 * standard adder methods *
 *                        *
 **************************/

/**
 * @brief adds an atom name to the atomNames vector
 *
 * @param atomName
 */
void MoleculeType::addAtomName(const std::string &atomName)
{
    _atomNames.push_back(atomName);
}

/**
 * @brief adds an external atom type to the externalAtomTypes vector
 *
 * @param externalAtomType
 */
void MoleculeType::addExternalAtomType(const size_t externalAtomType)
{
    _externalAtomTypes.push_back(externalAtomType);
}

/**
 * @brief adds a partial charge to the partialCharges vector
 *
 * @param partialCharge
 */
void MoleculeType::addPartialCharge(const double partialCharge)
{
    _partialCharges.push_back(partialCharge);
}

/**
 * @brief adds an external global VDW type to the externalGlobalVDWTypes vector
 *
 * @param externalGlobalVDWType
 */
void MoleculeType::addExternalGlobalVDWType(const size_t externalGlobalVDWType)
{
    _externalGlobalVDWTypes.push_back(externalGlobalVDWType);
}

/**
 * @brief adds an element to the externalToInternalAtomTypes map
 *
 * @param key
 * @param value
 */
void MoleculeType::addExternalToInternalAtomTypeElement(
    const size_t key,
    const size_t value
)
{
    _externalToInternalAtomTypes.try_emplace(key, value);
}

/**
 * @brief adds an atom type to the atomTypes vector
 *
 * @param atomType
 */
void MoleculeType::addAtomType(const size_t atomType)
{
    _atomTypes.push_back(atomType);
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief sets the name of the molecule
 *
 * @param name
 */
void MoleculeType::setName(const std::string_view &name) { _name = name; }

/**
 * @brief sets the number of atoms in the molecule
 *
 * @param numberOfAtoms
 */
void MoleculeType::setNumberOfAtoms(const size_t numberOfAtoms)
{
    _numberOfAtoms = numberOfAtoms;
}

/**
 * @brief sets the moltype of the molecule
 *
 * @param moltype
 */
void MoleculeType::setMoltype(const size_t moltype) { _moltype = moltype; }

/**
 * @brief sets the charge of the molecule
 *
 * @param charge
 */
void MoleculeType::setCharge(const double charge) { _charge = charge; }

/**
 * @brief sets the partial charge of an atom
 *
 * @param index
 * @param partialCharge
 */
void MoleculeType::setPartialCharge(
    const size_t index,
    const double partialCharge
)
{
    _partialCharges[index] = partialCharge;
}

/**
 * @brief sets the partial charges of the molecule
 *
 * @param partialCharges
 */
void MoleculeType::setPartialCharges(const std::vector<double> &partialCharges)
{
    _partialCharges = partialCharges;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the number of atoms in the molecule
 *
 * @return size_t
 */
size_t MoleculeType::getNumberOfAtoms() const { return _numberOfAtoms; }

/**
 * @brief get the moltype of the molecule
 *
 * @return size_t
 */
size_t MoleculeType::getMoltype() const { return _moltype; }

/**
 * @brief get the external atom type of an atom
 *
 * @param index
 * @return size_t
 */
size_t MoleculeType::getExternalAtomType(const size_t index) const
{
    return _externalAtomTypes[index];
}

/**
 * @brief get the atom type of an atom
 *
 * @param index
 * @return size_t
 */
size_t MoleculeType::getAtomType(const size_t index) const
{
    return _atomTypes[index];
}

/**
 * @brief get the internal atom type of an atom
 *
 * @param type
 * @return size_t
 */
size_t MoleculeType::getInternalAtomType(const size_t type) const
{
    return _externalToInternalAtomTypes.at(type);
}

/**
 * @brief get the charge of the molecule
 *
 * @return double
 */
double MoleculeType::getCharge() const { return _charge; }

/**
 * @brief get the partial charge of an atom
 *
 * @param index
 * @return double
 */
double MoleculeType::getPartialCharge(const size_t index) const
{
    return _partialCharges[index];
}

/**
 * @brief get the name of the molecule
 *
 * @return std::string
 */
std::string MoleculeType::getName() const { return _name; }

/**
 * @brief get the name of an atom
 *
 * @param index
 * @return std::string
 */
std::string MoleculeType::getAtomName(const size_t index) const
{
    return _atomNames[index];
}

/**
 * @brief get the atom names of the molecule
 *
 * @return std::vector<std::string>&
 */
std::vector<std::string> &MoleculeType::getAtomNames() { return _atomNames; }

/**
 * @brief get the external atom types of the molecule
 *
 * @return std::vector<size_t>&
 */
std::vector<size_t> &MoleculeType::getExternalAtomTypes()
{
    return _externalAtomTypes;
}

/**
 * @brief get the external global VDW types of the molecule
 *
 * @return std::vector<size_t>&
 */
std::vector<size_t> &MoleculeType::getExternalGlobalVDWTypes()
{
    return _externalGlobalVDWTypes;
}

/**
 * @brief get the partial charges of the molecule
 *
 * @return std::vector<double>&
 */
std::vector<double> &MoleculeType::getPartialCharges()
{
    return _partialCharges;
}

/**
 * @brief get the external to internal atom types of the molecule
 *
 * @return std::map<size_t, size_t>
 */
std::map<size_t, size_t> MoleculeType::getExternalToInternalAtomTypes() const
{
    return _externalToInternalAtomTypes;
}