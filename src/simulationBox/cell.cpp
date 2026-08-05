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

#include "cell.hpp"

#include "molecule.hpp"             // for Molecule
#include "simulationBox.hpp"        // for SimulationBox
#include "waterModelSettings.hpp"   // for WaterModelSettings

using namespace settings;
using namespace simulationBox;
using namespace linearAlgebra;

/**
 * @brief clears the molecules vector
 *
 */
void Cell::clearMolecules() { _molecules.clear(); }

/**
 * @brief clears the atoms vector
 *
 */
void Cell::clearAtoms() { _atoms.clear(); }

/**
 * @brief adds a molecule to the molecules vector
 *
 * @param molecule
 */
void Cell::addMolecule(Molecule &molecule) { _molecules.push_back(&molecule); }

/**
 * @brief adds a molecule to the molecules vector
 *
 * @param molecule
 */
void Cell::addMolecule(Molecule *molecule) { _molecules.push_back(molecule); }

/**
 * @brief adds a neighbour cell to the neighbourCells vector
 *
 * @param cell
 */
void Cell::addNeighbourCell(Cell *cell) { _neighbourCells.push_back(cell); }

/**
 * @brief adds atoms to the atoms vector
 *
 * @param lowerBoundary
 */
void Cell::addAtoms(const std::vector<Atom *> &atomPointers)
{
    _atoms.push_back(atomPointers);
}

/**
 * @brief Assign molecule indices to hybrid-zone and active/inactive buckets
 *
 * @details Uses the current `_molecules` order; call after molecules have been
 * added and hybrid zone have been assigned
 */
void Cell::assignMoleculeHybridZoneIndices()
{
    _coreMoleculeIndices.clear();
    _smoothingMoleculeIndices.clear();
    _nonSmoothingMoleculeIndices.clear();
    _activeMoleculeIndices.clear();
    _inactiveNonCoreMoleculeIndices.clear();

    using enum HybridZone;
    const auto nMol = getNumberOfMolecules();

    for (size_t mol = 0; mol < nMol; ++mol)
    {
        const auto hybridZone = _molecules[mol]->getHybridZone();
        const bool isCore     = (hybridZone == CORE);
        const auto isActive   = _molecules[mol]->isActive();

        if (hybridZone == CORE)
            _coreMoleculeIndices.push_back(mol);
        else if (hybridZone == SMOOTHING)
            _smoothingMoleculeIndices.push_back(mol);

        if (hybridZone != SMOOTHING)
            _nonSmoothingMoleculeIndices.push_back(mol);

        if (isActive)
            _activeMoleculeIndices.push_back(mol);
        else if (!isCore)
            _inactiveNonCoreMoleculeIndices.push_back(mol);
    }
}

/**
 * @brief assigns the indices of water molecules in the cell
 *
 * @param simBox
 */
void Cell::assignWaterMoleculeIndices(const SimulationBox &simBox)
{
    const auto isWaterInterModelSet =
        WaterModelSettings::isInterWaterModelSet();

    if (!isWaterInterModelSet)
        return;

    _waterMoleculeIndices.clear();

    const auto nMol           = getNumberOfMolecules();
    const auto waterTypeValue = simBox.getWaterType().value_or(size_t{0});

    for (size_t mol = 0; mol < nMol; ++mol)
    {
        const auto moltype = _molecules[mol]->getMoltype();

        if (moltype == waterTypeValue)
            _waterMoleculeIndices.push_back(mol);
    }
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief returns the number of molecules in the cell
 *
 * @return size_t
 */
size_t Cell::getNumberOfMolecules() const { return _molecules.size(); }

/**
 * @brief returns the number of neighbour cells
 *
 * @return size_t
 */
size_t Cell::getNumberOfNeighbourCells() const
{
    return _neighbourCells.size();
}

/**
 * @brief returns the lower boundary of the cell
 *
 * @return const Vec3D&
 */
const Vec3D &Cell::getLowerBoundary() const { return _lowerBoundary; }

/**
 * @brief returns the upper boundary of the cell
 *
 * @return const Vec3D&
 */
const Vec3D &Cell::getUpperBoundary() const { return _upperBoundary; }

/**
 * @brief returns the cell index
 *
 * @return const Vec3Dul&
 */
const Vec3Dul &Cell::getCellIndex() const { return _cellIndex; }

/**
 * @brief returns the molecules vector
 *
 * @return std::vector<Molecule*>&
 */
const std::vector<Molecule *> &Cell::getMolecules() const { return _molecules; }

/**
 * @brief returns the molecules vector
 *
 * @return std::vector<Molecule*>&
 */
std::vector<Molecule *> &Cell::getMolecules() { return _molecules; }

/**
 * @brief returns the neighbour cell at the given index
 *
 * @param index
 * @return Cell*
 */
Cell *Cell::getNeighbourCell(const size_t index) const
{
    return _neighbourCells[index];
}

/**
 * @brief returns the neighbour cells vector
 *
 * @return const std::vector<Cell*>&
 */
const std::vector<Cell *> &Cell::getNeighbourCells() const
{
    return _neighbourCells;
}

/**
 * @brief returns the molecule indices in the core hybrid zone
 *
 * @return const std::vector<size_t>&
 */
const std::vector<size_t> &Cell::getCoreMoleculeIndices() const
{
    return _coreMoleculeIndices;
}

/**
 * @brief returns the molecule indices in the smoothing hybrid zone
 *
 * @return const std::vector<size_t>&
 */
const std::vector<size_t> &Cell::getSmoothingMoleculeIndices() const
{
    return _smoothingMoleculeIndices;
}

/**
 * @brief returns the molecule indices outside the smoothing hybrid zone
 *
 * @return const std::vector<size_t>&
 */
const std::vector<size_t> &Cell::getNonSmoothingMoleculeIndices() const
{
    return _nonSmoothingMoleculeIndices;
}

/**
 * @brief returns the indices of the active molecules
 *
 * @return const std::vector<size_t>&
 */
const std::vector<size_t> &Cell::getActiveMoleculeIndices() const
{
    return _activeMoleculeIndices;
}

/**
 * @brief returns the indices of the inactive molecules
 *
 * @return const std::vector<size_t>&
 */
const std::vector<size_t> &Cell::getInactiveNonCoreMoleculeIndices() const
{
    return _inactiveNonCoreMoleculeIndices;
}

/**
 * @brief returns the indices of the water molecules
 *
 * @return const std::vector<size_t>&
 */
const std::vector<size_t> &Cell::getWaterMoleculeIndices() const
{
    return _waterMoleculeIndices;
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the lower boundary of the cell
 *
 * @param lowerBoundary
 */
void Cell::setLowerBoundary(const Vec3D &lowerBoundary)
{
    _lowerBoundary = lowerBoundary;
}

/**
 * @brief set the upper boundary of the cell
 *
 * @param upperBoundary
 */
void Cell::setUpperBoundary(const Vec3D &upperBoundary)
{
    _upperBoundary = upperBoundary;
}

/**
 * @brief set the cell index
 *
 * @param cellIndex
 */
void Cell::setCellIndex(const Vec3Dul &cellIndex) { _cellIndex = cellIndex; }