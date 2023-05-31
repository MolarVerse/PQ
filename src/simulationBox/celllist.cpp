#include "celllist.hpp"
#include "simulationBox.hpp"
#include "cell.hpp"

#include <cmath>
#include <iostream>

using namespace std;

void CellList::setup(const SimulationBox &simulationBox)
{
    determineCellSize(simulationBox);

    _cells.resize(_nCellsX * _nCellsY * _nCellsZ);

    determineCellBoundaries(simulationBox);

    addNeighbouringCells(simulationBox);
}

void CellList::determineCellSize(const SimulationBox &simulationBox)
{
    auto boxDimensions = simulationBox._box.getBoxDimensions();
    _cellSize = {boxDimensions[0] / static_cast<double>(_nCellsX), boxDimensions[1] / static_cast<double>(_nCellsY), boxDimensions[2] / static_cast<double>(_nCellsZ)};
}

void CellList::determineCellBoundaries(const SimulationBox &simulationBox)
{
    for (int i = 0; i < _nCellsX; ++i)
    {
        for (int j = 0; j < _nCellsY; ++j)
        {
            for (int k = 0; k < _nCellsZ; ++k)
            {
                const auto cellIndex = i * _nCellsY * _nCellsZ + j * _nCellsZ + k;
                auto *cell = &_cells[cellIndex];

                auto boxDimensions = simulationBox._box.getBoxDimensions();

                auto lowerBoundary = {-boxDimensions[0] / 2.0 + i * _cellSize[0], -boxDimensions[1] / 2.0 + j * _cellSize[1], -boxDimensions[2] / 2.0 + k * _cellSize[2]};
                auto upperBoundary = {-boxDimensions[0] / 2.0 + (i + 1) * _cellSize[0], -boxDimensions[1] / 2.0 + (j + 1) * _cellSize[1], -boxDimensions[2] / 2.0 + (k + 1) * _cellSize[2]};

                cell->setLowerBoundary(lowerBoundary);
                cell->setUpperBoundary(upperBoundary);

                cell->setCellIndex({i, j, k});
            }
        }
    }
}

void CellList::addNeighbouringCells(const SimulationBox &simulationBox)
{
    _nNeighbourCellsX = static_cast<size_t>(ceil(simulationBox.getRcCutOff() / _cellSize[0]));
    _nNeighbourCellsY = static_cast<size_t>(ceil(simulationBox.getRcCutOff() / _cellSize[1]));
    _nNeighbourCellsZ = static_cast<size_t>(ceil(simulationBox.getRcCutOff() / _cellSize[2]));

    for (auto &cell : _cells)
    {
        addCellPointers(cell);
    }
}

void CellList::addCellPointers(Cell &cell)
{
    const size_t totalCellNeighbours = (_nNeighbourCellsX * 2 + 1) * (_nNeighbourCellsY * 2 + 1) * (_nNeighbourCellsZ * 2 + 1);

    for (int i = -_nNeighbourCellsX; i <= _nNeighbourCellsX; ++i)
    {
        for (int j = -_nNeighbourCellsY; j <= _nNeighbourCellsY; ++j)
        {
            for (int k = -_nNeighbourCellsZ; k <= _nNeighbourCellsZ; ++k)
            {
                if ((i == 0) && (j == 0) && (k == 0))
                    continue;

                int neighbourCellIndexX = i + cell.getCellIndex()[0];
                int neighbourCellIndexY = j + cell.getCellIndex()[1];
                int neighbourCellIndexZ = k + cell.getCellIndex()[2];

                neighbourCellIndexX -= _nCellsX * static_cast<size_t>(floor(neighbourCellIndexX / static_cast<double>(_nCellsX)));
                neighbourCellIndexY -= _nCellsY * static_cast<size_t>(floor(neighbourCellIndexY / static_cast<double>(_nCellsY)));
                neighbourCellIndexZ -= _nCellsZ * static_cast<size_t>(floor(neighbourCellIndexZ / static_cast<double>(_nCellsZ)));

                const auto neighbourCellIndex = neighbourCellIndexX * _nCellsY * _nCellsZ + neighbourCellIndexY * _nCellsZ + neighbourCellIndexZ;

                Cell *neighbourCell = &_cells[neighbourCellIndex];

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

        _cells.resize(_nCellsX * _nCellsY * _nCellsZ);

        determineCellBoundaries(simulationBox);

        addNeighbouringCells(simulationBox);
    }

    for (auto &cell : _cells)
    {
        cell.clearMolecules();
        cell.clearAtomIndices();
    }

    vector<double> position(3);

    const size_t numberOfMolecules = simulationBox.getNumberOfMolecules();

    for (size_t i = 0; i < numberOfMolecules; ++i)
    {
        Molecule *molecule = &simulationBox._molecules[i];
        auto mapCellIndexToAtomIndex = map<size_t, std::vector<size_t>>();
        const size_t numberOfAtoms = molecule->getNumberOfAtoms();

        for (size_t j = 0; j < numberOfAtoms; ++j)
        {
            molecule->getAtomPositions(j, position);

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

vector<size_t> CellList::getCellIndexOfMolecule(const SimulationBox &simulationBox, const vector<double> &position)
{
    const auto boxDimensions = simulationBox._box.getBoxDimensions();

    auto cellIndexX = static_cast<size_t>(floor((position[0] + boxDimensions[0] / 2.0) / _cellSize[0]));
    auto cellIndexY = static_cast<size_t>(floor((position[1] + boxDimensions[1] / 2.0) / _cellSize[1]));
    auto cellIndexZ = static_cast<size_t>(floor((position[2] + boxDimensions[2] / 2.0) / _cellSize[2]));

    cellIndexX -= _nCellsX * static_cast<size_t>(floor(static_cast<double>(cellIndexX) / static_cast<double>(_nCellsX)));
    cellIndexY -= _nCellsY * static_cast<size_t>(floor(static_cast<double>(cellIndexY) / static_cast<double>(_nCellsY)));
    cellIndexZ -= _nCellsZ * static_cast<size_t>(floor(static_cast<double>(cellIndexZ) / static_cast<double>(_nCellsZ)));

    return {cellIndexX, cellIndexY, cellIndexZ};
}
