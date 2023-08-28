#include "celllist.hpp"

#include "cell.hpp"            // for Cell
#include "exceptions.hpp"      // for CellListException
#include "molecule.hpp"        // for Molecule
#include "simulationBox.hpp"   // for SimulationBox

#include <algorithm>     // for ranges::for_each
#include <functional>    // for identity
#include <map>           // for map
#include <string_view>   // for string_view

using namespace simulationBox;
using namespace linearAlgebra;

/**
 * @brief get linearized cell index
 *
 * @param cellIndices
 * @return size_t
 */
[[nodiscard]] size_t CellList::getCellIndex(const Vec3Dul &cellIndices) const
{
    return cellIndices[0] * _nCells[1] * _nCells[2] + cellIndices[1] * _nCells[2] + cellIndices[2];
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

    checkCoulombCutoff(simulationBox.getCoulombRadiusCutOff());

    determineCellBoundaries(simulationBox.getBoxDimensions());

    addNeighbouringCells(simulationBox.getCoulombRadiusCutOff());
}

/**
 * @brief determine cell size
 *
 * @param simulationBox
 */
void CellList::determineCellSize(const linearAlgebra::Vec3D &box) { _cellSize = box / Vec3D(_nCells); }

/**
 * @brief check if coulomb cutoff is smaller than half of the largest cell size
 *
 * @throws customException::CellListException if coulomb cutoff is smaller than half of the largest cell size
 *
 * @param coulombCutoff
 */
void CellList::checkCoulombCutoff(const double coulombCutoff) const
{
    if (coulombCutoff < maximum(_cellSize) / 2.0)
        throw customException::CellListException("Coulomb cutoff is smaller than half of the largest cell size.");
}

/**
 * @brief determine cell boundaries
 *
 * @param simulationBox
 */
void CellList::determineCellBoundaries(const linearAlgebra::Vec3D &box)
{
    for (size_t i = 0; i < _nCells[0]; ++i)
        for (size_t j = 0; j < _nCells[1]; ++j)
            for (size_t k = 0; k < _nCells[2]; ++k)
            {
                const auto ijk       = Vec3Dul(i, j, k);
                const auto cellIndex = getCellIndex(ijk);
                auto      *cell      = &_cells[cellIndex];

                cell->setLowerBoundary(-box / 2.0 + Vec3D(ijk) * _cellSize);
                cell->setUpperBoundary(-box / 2.0 + (Vec3D(ijk) + 1) * _cellSize);

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

    std::ranges::for_each(_cells, [this](auto &cell) { addNeighbouringCellPointers(cell); });
}

/**
 * @brief add neighbouring cell pointers to a cell
 *
 * @param cell
 */
void CellList::addNeighbouringCellPointers(Cell &cell)
{
    const size_t totalCellNeighbours = prod(_nNeighbourCells * 2 + 1);

    for (int i = -int(_nNeighbourCells[0]); i <= int(_nNeighbourCells[0]); ++i)
        for (int j = -int(_nNeighbourCells[1]); j <= int(_nNeighbourCells[1]); ++j)
            for (int k = -int(_nNeighbourCells[2]); k <= int(_nNeighbourCells[2]); ++k)
            {
                const auto ijk = Vec3Di(i, j, k);

                if (ijk == Vec3Di(0, 0, 0))
                    continue;

                auto neighbourCellIndex  = ijk + Vec3Di(cell.getCellIndex());
                neighbourCellIndex      -= Vec3Di(_nCells) * Vec3Di(floor(Vec3D(neighbourCellIndex) / Vec3D(_nCells)));

                const auto neighbourCellIndexScalar = getCellIndex(Vec3Dul(neighbourCellIndex));

                Cell *neighbourCell = &_cells[neighbourCellIndexScalar];

                cell.addNeighbourCell(neighbourCell);

                if (cell.getNumberOfNeighbourCells() == (totalCellNeighbours - 1) / 2)
                    return;
            }
}

/**
 * @brief update cell list after during simulation
 *
 * @details it checks if the box size has changed and if so it clears the cell list and sets it up again
 * then it clears all molecular and atomic information in the cells (for the case the box size has not changed)
 * and add the molecules and atomic indices again to the cells depending on their new positions
 *
 * @param simulationBox
 */
void CellList::updateCellList(SimulationBox &simulationBox)
{
    if (!_activated)
        return;

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
}

/**
 * @brief add molecules and atom indices to cells
 *
 * @details it is not sufficient to just add the molecules to the cells, because this program works on an atom based cutoff scheme
 * therefore e.g. the center of mass of a molecule could be in one cell, but some of its atoms could be in a neighbouring cell
 *
 * @param simulationBox
 */
void CellList::addMoleculesToCells(SimulationBox &simulationBox)
{
    const auto box = simulationBox.getBoxDimensions();

    for (size_t i = 0, nMolecules = simulationBox.getNumberOfMolecules(); i < nMolecules; ++i)
    {
        auto *molecule                = &simulationBox.getMolecule(i);
        auto  mapCellIndexToAtomIndex = std::map<size_t, std::vector<size_t>>();

        for (size_t j = 0, nAtoms = molecule->getNumberOfAtoms(); j < nAtoms; ++j)
        {
            const auto position = molecule->getAtomPosition(j);

            const auto atomCellIndices = getCellIndexOfAtom(box, position);
            const auto cellIndexScalar = getCellIndex(atomCellIndices);

            const auto &[_, successful] = mapCellIndexToAtomIndex.try_emplace(cellIndexScalar, std::vector<size_t>({j}));

            if (!successful)
                mapCellIndexToAtomIndex[cellIndexScalar].push_back(j);
        }

        auto addMoleculeAndAtomIndicesToCell = [this, molecule](auto &pair)
        {
            const auto &[cellIndex, atomIndices] = pair;
            _cells[cellIndex].addMolecule(molecule);
            _cells[cellIndex].addAtomIndices(atomIndices);
        };

        std::ranges::for_each(mapCellIndexToAtomIndex, addMoleculeAndAtomIndicesToCell);
    }
}

/**
 * @brief get cell index of atom
 *
 * @param simulationBox
 * @param position
 * @return Vec3Dul
 */
[[nodiscard]] Vec3Dul CellList::getCellIndexOfAtom(const Vec3D &box, const Vec3D &position) const
{
    auto cellIndex = Vec3Dul(floor((position + box / 2.0) / _cellSize));

    cellIndex -= _nCells * Vec3Dul(floor(Vec3D(cellIndex) / Vec3D(_nCells)));

    return cellIndex;
}