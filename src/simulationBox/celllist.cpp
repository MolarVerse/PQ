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
    _cellSize = {boxDimensions[0] / _nCellsX, boxDimensions[1] / _nCellsY, boxDimensions[2] / _nCellsZ};
}

void CellList::determineCellBoundaries(const SimulationBox &simulationBox)
{
    for (int i = 0; i < _nCellsX; i++)
    {
        for (int j = 0; j < _nCellsY; j++)
        {
            for (int k = 0; k < _nCellsZ; k++)
            {
                auto cellIndex = i * _nCellsY * _nCellsZ + j * _nCellsZ + k;
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
    _nNeighbourCellsX = int(ceil(simulationBox.getRcCutOff() / _cellSize[0]));
    _nNeighbourCellsY = int(ceil(simulationBox.getRcCutOff() / _cellSize[1]));
    _nNeighbourCellsZ = int(ceil(simulationBox.getRcCutOff() / _cellSize[2]));

    for (auto &cell : _cells)
    {
        addCellPointers(cell);
    }
}

void CellList::addCellPointers(Cell &cell)
{
    int totalCellNeighbours = (_nNeighbourCellsX * 2 + 1) * (_nNeighbourCellsY * 2 + 1) * (_nNeighbourCellsZ * 2 + 1);
    for (int i = -_nNeighbourCellsX; i <= _nNeighbourCellsX; i++)
    {
        for (int j = -_nNeighbourCellsY; j <= _nNeighbourCellsY; j++)
        {
            for (int k = -_nNeighbourCellsZ; k <= _nNeighbourCellsZ; k++)
            {
                if (i == 0 && j == 0 && k == 0)
                    continue;

                int neighbourCellIndexX = i + cell.getCellIndex()[0];
                int neighbourCellIndexY = j + cell.getCellIndex()[1];
                int neighbourCellIndexZ = k + cell.getCellIndex()[2];

                neighbourCellIndexX -= _nCellsX * int(floor((double)neighbourCellIndexX / (double)_nCellsX));
                neighbourCellIndexY -= _nCellsY * int(floor((double)neighbourCellIndexY / (double)_nCellsY));
                neighbourCellIndexZ -= _nCellsZ * int(floor((double)neighbourCellIndexZ / (double)_nCellsZ));

                int neighbourCellIndex = neighbourCellIndexX * _nCellsY * _nCellsZ + neighbourCellIndexY * _nCellsZ + neighbourCellIndexZ;

                Cell *neighbourCell = &_cells[neighbourCellIndex];

                cell.addNeighbourCell(neighbourCell);

                if (cell.getNeighbourCellSize() == int((totalCellNeighbours - 1) / 2))
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

    for (int i = 0; i < simulationBox.getNumberOfMolecules(); i++)
    {
        Molecule *molecule = &simulationBox._molecules[i];
        auto mapCellIndexToAtomIndex = map<int, std::vector<int>>();

        for (int j = 0; j < molecule->getNumberOfAtoms(); j++)
        {
            molecule->getAtomPositions(j, position);

            auto atomCellIndices = getCellIndexOfMolecule(simulationBox, position);
            auto cellIndex = getCellIndex(atomCellIndices);

            const auto &[_, successful] = mapCellIndexToAtomIndex.try_emplace(cellIndex, vector<int>({j}));
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

vector<int> CellList::getCellIndexOfMolecule(const SimulationBox &simulationBox, const vector<double> &position)
{
    auto boxDimensions = simulationBox._box.getBoxDimensions();

    auto cellIndexX = int(floor((position[0] + boxDimensions[0] / 2.0) / _cellSize[0]));
    auto cellIndexY = int(floor((position[1] + boxDimensions[1] / 2.0) / _cellSize[1]));
    auto cellIndexZ = int(floor((position[2] + boxDimensions[2] / 2.0) / _cellSize[2]));

    cellIndexX -= _nCellsX * int(floor((double)cellIndexX / (double)_nCellsX));
    cellIndexY -= _nCellsY * int(floor((double)cellIndexY / (double)_nCellsY));
    cellIndexZ -= _nCellsZ * int(floor((double)cellIndexZ / (double)_nCellsZ));

    return {cellIndexX, cellIndexY, cellIndexZ};
}
