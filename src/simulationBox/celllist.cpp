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

#include "celllist.hpp"

#include <algorithm>     // for ranges::for_each
#include <functional>    // for identity
#include <map>           // for map
#include <string_view>   // for string_view

#include "cell.hpp"                // for Cell
#include "exceptions.hpp"          // for CellListException
#include "molecule.hpp"            // for Molecule
#include "potentialSettings.hpp"   // for PotentialSettings
#include "simulationBox.hpp"       // for SimulationBox

using namespace simulationBox;
using namespace settings;
using namespace linearAlgebra;
using namespace customException;

/**
 * @brief clone cell list
 *
 * @return std::shared_ptr<CellList>
 */
std::shared_ptr<CellList> CellList::clone() const
{
    return std::make_shared<CellList>(*this);
}

/**
 * @brief get linearized cell index
 *
 * @param cellIndices
 * @return size_t
 */
size_t CellList::getCellIndex(const Vec3Dul &cellIndices) const
{
    const auto outerProduct = cellIndices[0] * _nCells[1] * _nCells[2];

    return outerProduct + cellIndices[1] * _nCells[2] + cellIndices[2];
}

/**
 * @brief setup cell list
 *
 * @details following steps are preformed:
 * 1) determine the cell size
 * 2) check if coulomb cutoff is smaller than half of the largest cell size
 * 3) determine the cell boundaries
 * 4) add neighbouring cells
 *
 * @param simulationBox
 */
void CellList::setup(const SimulationBox &simulationBox)
{
    determineCellSize(simulationBox.getBoxDimensions());

    checkCoulombCutoff(PotentialSettings::getCoulombRadiusCutOff());

    determineCellBoundaries(simulationBox.getBoxDimensions());

    addNeighbouringCells(PotentialSettings::getCoulombRadiusCutOff());
}

/**
 * @brief determine cell size
 *
 * @param simulationBox
 */
void CellList::determineCellSize(const Vec3D &box)
{
    _cellSize = box / Vec3D(_nCells);
}

/**
 * @brief check if coulomb cutoff is smaller than half of the largest cell size
 *
 * @throws customException::CellListException if coulomb cutoff is smaller than
 * half of the largest cell size
 *
 * @param coulombCutoff
 */
void CellList::checkCoulombCutoff(const double coulombCutoff) const
{
    if (coulombCutoff < maximum(_cellSize) / 2.0)
        throw CellListException(
            "Coulomb cutoff is smaller than half of the largest cell size."
        );
}

/**
 * @brief determine cell boundaries
 *
 * @param simulationBox
 */
void CellList::determineCellBoundaries(const Vec3D &box)
{
    for (size_t i = 0; i < _nCells[0]; ++i)
        for (size_t j = 0; j < _nCells[1]; ++j)
            for (size_t k = 0; k < _nCells[2]; ++k)
            {
                const auto ijk       = Vec3Dul(i, j, k);
                const auto cellIndex = getCellIndex(ijk);
                auto      *cell      = &_cells[cellIndex];

                cell->setLowerBoundary(-box / 2.0 + Vec3D(ijk) * _cellSize);
                cell->setUpperBoundary(
                    -box / 2.0 + (Vec3D(ijk) + 1) * _cellSize
                );

                cell->setCellIndex(ijk);
            }
}

/**
 * @brief add neighbouring cells
 *
 * @param simulationBox
 */
void CellList::addNeighbouringCells(const double coulombCutoff)
{
    _nNeighbourCells = Vec3Dul(ceil(coulombCutoff / _cellSize));

    auto addCell = [this](auto &cell) { addNeighbouringCellPointers(cell); };

    std::ranges::for_each(_cells, addCell);
}

/**
 * @brief add neighbouring cell pointers to a cell
 *
 * @param cell
 */
void CellList::addNeighbouringCellPointers(Cell &cell)
{
    const size_t totalCellNeighbours = prod(_nNeighbourCells * 2 + 1);

    const auto nNeighCells0 = int(_nNeighbourCells[0]);
    const auto nNeighCells1 = int(_nNeighbourCells[1]);
    const auto nNeighCells2 = int(_nNeighbourCells[2]);

    for (int i = -nNeighCells0; i <= nNeighCells0; ++i)
        for (int j = -nNeighCells1; j <= nNeighCells1; ++j)
            for (int k = -nNeighCells2; k <= nNeighCells2; ++k)
            {
                const auto ijk = Vec3Di(i, j, k);

                if (ijk == Vec3Di(0, 0, 0))
                    continue;

                auto neighCellIndex  = ijk + Vec3Di(cell.getCellIndex());
                auto indices         = Vec3D(neighCellIndex) / Vec3D(_nCells);
                indices              = floor(indices);
                neighCellIndex      -= Vec3Di(_nCells) * Vec3Di(indices);

                const auto scalarIndex          = Vec3Dul(neighCellIndex);
                const auto neighCellIndexScalar = getCellIndex(scalarIndex);

                Cell *neighbourCell = &_cells[neighCellIndexScalar];

                cell.addNeighbourCell(neighbourCell);

                const auto nNeighCells = cell.getNumberOfNeighbourCells();

                if (nNeighCells == (totalCellNeighbours - 1) / 2)
                    return;
            }
}

/**
 * @brief update cell list after during simulation
 *
 * @details it checks if the box size has changed and if so it clears the cell
 * list and sets it up again then it clears all molecular and atomic information
 * in the cells (for the case the box size has not changed) and add the
 * molecules and atomic indices again to the cells depending on their new
 * positions
 *
 * @param simulationBox
 */
void CellList::updateCellList(SimulationBox &simulationBox)
{
    if (!_activated)
        return;

    startTimingsSection("Update");

    if (simulationBox.getBoxSizeHasChanged())
    {
        _cells.clear();
        resizeCells();
        setup(simulationBox);
    }

    auto clearMoleculesAndAtomIndices = [](auto &cell)
    {
        cell.clearMolecules();
        cell.clearAtomIndices();
    };

    std::ranges::for_each(_cells, clearMoleculesAndAtomIndices);

    addMoleculesToCells(simulationBox);

    stopTimingsSection("Update");
}

/**
 * @brief add molecules and atom indices to cells
 *
 * @details it is not sufficient to just add the molecules to the cells, because
 * this program works on an atom based cutoff scheme therefore e.g. the center
 * of mass of a molecule could be in one cell, but some of its atoms could be in
 * a neighbouring cell
 *
 * @param simulationBox
 */
void CellList::addMoleculesToCells(SimulationBox &simulationBox)
{
    const auto box        = simulationBox.getBoxDimensions();
    const auto nMolecules = simulationBox.getNumberOfMolecules();

    for (size_t i = 0; i < nMolecules; ++i)
    {
        auto *molecule                = &simulationBox.getMolecule(i);
        auto  mapCellIndexToAtomIndex = std::map<size_t, std::vector<size_t>>();

        const auto nAtomsInMolecule = molecule->getNumberOfAtoms();

        for (size_t j = 0; j < nAtomsInMolecule; ++j)
        {
            const auto position = molecule->getAtomPosition(j);

            const auto atomCellIndices = getCellIndexOfAtom(box, position);
            const auto cellIndexScalar = getCellIndex(atomCellIndices);

            const auto &[_, successful] = mapCellIndexToAtomIndex.try_emplace(
                cellIndexScalar,
                std::vector<size_t>({j})
            );

            if (!successful)
                mapCellIndexToAtomIndex[cellIndexScalar].push_back(j);
        }

        auto addMoleculeAndAtomIndicesToCell = [this, molecule](auto &pair)
        {
            const auto &[cellIndex, atomIndices] = pair;
            _cells[cellIndex].addMolecule(molecule);
            _cells[cellIndex].addAtomIndices(atomIndices);
        };

        std::ranges::for_each(
            mapCellIndexToAtomIndex,
            addMoleculeAndAtomIndicesToCell
        );
    }
}

/**
 * @brief get cell index of atom
 *
 * @param simulationBox
 * @param position
 * @return Vec3Dul
 */
Vec3Dul CellList::getCellIndexOfAtom(
    const Vec3D &box,
    const Vec3D &position
) const
{
    auto cellIndex = Vec3Dul(floor((position + box / 2.0) / _cellSize));

    cellIndex -= _nCells * Vec3Dul(floor(Vec3D(cellIndex) / Vec3D(_nCells)));

    return cellIndex;
}

/**
 * @brief resize cells
 *
 */
void CellList::resizeCells() { _cells.resize(prod(_nCells)); }

/**
 * @brief add cell to cell list
 *
 * @param cell
 */
void CellList::addCell(const Cell &cell) { _cells.push_back(cell); }

/*****************************
 *                           *
 * standard activate methods *
 *                           *
 *****************************/

/**
 * @brief activate cell list
 *
 */
void CellList::activate() { _activated = true; }

/**
 * @brief deactivate cell list
 *
 */
void CellList::deactivate() { _activated = false; }

/**
 * @brief check if cell list is active
 *
 * @return true
 * @return false
 */
bool CellList::isActive() const { return _activated; }

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get number of cells
 *
 * @return Vec3Dul
 */
Vec3Dul CellList::getNumberOfCells() const { return _nCells; }

/**
 * @brief get number of neighbour cells
 *
 * @return Vec3Dul
 */
Vec3Dul CellList::getNumberOfNeighbourCells() const { return _nNeighbourCells; }

/**
 * @brief get cell size
 *
 * @return Vec3D
 */
Vec3D CellList::getCellSize() const { return _cellSize; }

/**
 * @brief get cells
 *
 * @return std::vector<Cell>
 */
std::vector<Cell> CellList::getCells() const { return _cells; }

/**
 * @brief get cell by index
 *
 * @param index
 * @return Cell&
 */
Cell &CellList::getCell(const size_t index) { return _cells[index]; }

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set number of cells
 *
 * @param nCells
 */
void CellList::setNumberOfCells(const size_t nCells)
{
    _nCells = {nCells, nCells, nCells};
}

/**
 * @brief set number of neighbour cells
 *
 * @param nCells
 */
void CellList::setNumberOfNeighbourCells(const size_t nCells)
{
    _nNeighbourCells = Vec3Dul(nCells);
}