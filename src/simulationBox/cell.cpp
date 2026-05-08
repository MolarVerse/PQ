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
void Cell::addAtoms(const std::vector<pq::Atom *> &atomPointers)
{
    _atoms.push_back(atomPointers);
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
 * @brief returns the molecule at the given index
 *
 * @param index
 * @return Molecule*
 */
Molecule *Cell::getMolecule(const size_t index) const
{
    return _molecules[index];
}

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
 * @return std::vector<Cell*>
 */
std::vector<Cell *> Cell::getNeighbourCells() const { return _neighbourCells; }

/**
 * @brief returns the atoms at the given molecule index
 *
 * @param index
 * @return const std::vector<size_t>&
 */
const std::vector<pq::Atom *> &Cell::getAtoms(const size_t molIndex) const
{
    return _atoms[molIndex];
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