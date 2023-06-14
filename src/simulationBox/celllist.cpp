#include "celllist.hpp"
#include "simulationBox.hpp"
#include "cell.hpp"

#include <cmath>
#include <iostream>

using namespace std;

void CellList::setup(const SimulationBox &simulationBox)
{
    determineCellSize(simulationBox);

    _cells.resize(prod(_nCells));

    determineCellBoundaries(simulationBox);

    addNeighbouringCells(simulationBox);
}

void CellList::determineCellSize(const SimulationBox &simulationBox)
{
    const auto box = simulationBox.getBoxDimensions();
    _cellSize = box / static_cast<Vec3D>(_nCells);
}

void CellList::determineCellBoundaries(const SimulationBox &simulationBox)
{
    for (size_t i = 0; i < _nCells[0]; ++i)
    {
        for (size_t j = 0; j < _nCells[1]; ++j)
        {
            for (size_t k = 0; k < _nCells[2]; ++k)
            {

                const auto cellIndex = i * _nCells[1] * _nCells[2] + j * _nCells[2] + k;
                auto *cell = &_cells[cellIndex];

                const auto box = simulationBox.getBoxDimensions();

                const auto ijk = Vec3Dul(i, j, k);

                const auto lowerBoundary = -box / 2.0 + static_cast<Vec3D>(ijk) * _cellSize;
                const auto upperBoundary = -box / 2.0 + static_cast<Vec3D>((ijk + 1)) * _cellSize;

                cell->setLowerBoundary(lowerBoundary);
                cell->setUpperBoundary(upperBoundary);

                cell->setCellIndex(ijk);
            }
        }
    }
}

void CellList::addNeighbouringCells(const SimulationBox &simulationBox)
{

    _nNeighbourCells = static_cast<Vec3Dul>(ceil(simulationBox.getRcCutOff() / _cellSize));

    for (auto &cell : _cells)
        addCellPointers(cell);
}

void CellList::addCellPointers(Cell &cell)
{
    const size_t totalCellNeighbours = prod(_nNeighbourCells * 2 + 1);

    int i0 = -_nNeighbourCells[0];
    int i1 = _nNeighbourCells[0];
    int j0 = -_nNeighbourCells[1];
    int j1 = _nNeighbourCells[1];
    int k0 = -_nNeighbourCells[2];
    int k1 = _nNeighbourCells[2];

    for (int i = i0; i <= i1; ++i)
    {
        for (int j = j0; j <= j1; ++j)
        {
            for (int k = k0; k <= k1; ++k)
            {
                const auto ijk = Vec3Di(i, j, k);

                if (ijk == Vec3Di(0, 0, 0))
                    continue;

                auto neighbourCellIndex = ijk + static_cast<Vec3Di>(cell.getCellIndex());
                neighbourCellIndex -= static_cast<Vec3Di>(_nCells) * static_cast<Vec3Di>(floor(static_cast<Vec3D>(neighbourCellIndex) / static_cast<Vec3D>(_nCells)));

                const auto neighbourCellIndexScalar = neighbourCellIndex[0] * _nCells[1] * _nCells[2] + neighbourCellIndex[1] * _nCells[2] + neighbourCellIndex[2];

                Cell *neighbourCell = &_cells[neighbourCellIndexScalar];

                cell.addNeighbourCell(neighbourCell);

                if (cell.getNeighbourCellSize() == (totalCellNeighbours - 1) / 2)
                    return;
            }
        }
    }
}

void CellList::updateCellList(SimulationBox &simulationBox)
{
    if (!_activated)
        return;

    if (simulationBox._box.getBoxSizeHasChanged())
    {
        _cells.clear();

        determineCellSize(simulationBox);

        _cells.resize(prod(_nCells));

        determineCellBoundaries(simulationBox);

        addNeighbouringCells(simulationBox);
    }

    for (auto &cell : _cells)
    {
        cell.clearMolecules();
        cell.clearAtomIndices();
    }

    Vec3D position(0.0, 0.0, 0.0);

    const size_t numberOfMolecules = simulationBox.getNumberOfMolecules();

    for (size_t i = 0; i < numberOfMolecules; ++i)
    {
        auto *molecule = &simulationBox.getMolecule(i);
        auto mapCellIndexToAtomIndex = map<size_t, std::vector<size_t>>();
        const size_t numberOfAtoms = molecule->getNumberOfAtoms();

        for (size_t j = 0; j < numberOfAtoms; ++j)
        {
            position = molecule->getAtomPosition(j);

            const auto atomCellIndices = getCellIndexOfMolecule(simulationBox, position);
            const auto cellIndex = getCellIndex(atomCellIndices);

            const auto &[_, successful] = mapCellIndexToAtomIndex.try_emplace(cellIndex, vector<size_t>({j}));
            if (!successful)
                mapCellIndexToAtomIndex[cellIndex].push_back(j);
        }

        for (const auto &[cellIndex, atomIndices] : mapCellIndexToAtomIndex)
        {
            _cells[cellIndex].addMolecule(molecule);
            _cells[cellIndex].addAtomIndices(atomIndices);
        }
    }
}

Vec3Dul CellList::getCellIndexOfMolecule(const SimulationBox &simulationBox, const Vec3D &position)
{
    const auto boxDimensions = simulationBox._box.getBoxDimensions();

    auto cellIndex = static_cast<Vec3Dul>(floor((position + boxDimensions / 2.0) / _cellSize));

    cellIndex -= _nCells * static_cast<Vec3Dul>(floor(static_cast<Vec3D>(cellIndex) / static_cast<Vec3D>(_nCells)));

    return cellIndex;
}