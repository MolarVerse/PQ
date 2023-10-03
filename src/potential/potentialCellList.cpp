/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "cell.hpp"           // for Cell, simulationBox
#include "celllist.hpp"       // for CellList
#include "physicalData.hpp"   // for PhysicalData
#include "potential.hpp"
#include "simulationBox.hpp"   // for SimulationBox

#include <cstddef>   // for size_t
#include <vector>    // for vector

using namespace potential;

/**
 * @brief calculates forces, coulombic and non-coulombic energy for cell list routine
 *
 * @details first loops over all possible combinations of molecules within the same cell, then over all possible molecule
 * combinations between adjacent cells. For the second loop over different cells, it is necessary to check if the two
 * molecules are the same to avoid double counting. Due to the cutoff criterion which is based on atoms a molecule can be
 * found in more than only one cell.
 *
 * @param simBox
 * @param physicalData
 * @param cellList
 */
inline void PotentialCellList::calculateForces(simulationBox::SimulationBox &simBox,
                                               physicalData::PhysicalData   &physicalData,
                                               simulationBox::CellList      &cellList)
{
    const auto box = simBox.getBoxPtr();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    for (const auto &cell1 : cellList.getCells())
    {
        const auto numberOfMolecules = cell1.getNumberOfMolecules();

        for (size_t mol1 = 0; mol1 < numberOfMolecules; ++mol1)
        {
            auto *molecule1 = cell1.getMolecule(mol1);

            for (size_t mol2 = 0; mol2 < mol1; ++mol2)
            {
                auto *molecule2 = cell1.getMolecule(mol2);

                for (const size_t atom1 : cell1.getAtomIndices(mol1))
                {
                    for (const size_t atom2 : cell1.getAtomIndices(mol2))
                    {
                        const auto [coulombEnergy, nonCoulombEnergy] =
                            calculateSingleInteraction(*box, *molecule1, *molecule2, atom1, atom2);

                        totalCoulombEnergy    += coulombEnergy;
                        totalNonCoulombEnergy += nonCoulombEnergy;
                    }
                }
            }
        }
    }
    for (const auto &cell1 : cellList.getCells())
    {
        const auto numberOfMoleculesInCell_i = cell1.getNumberOfMolecules();

        for (const auto *cell2 : cell1.getNeighbourCells())
        {
            const auto numberOfMoleculesInCell_j = cell2->getNumberOfMolecules();

            for (size_t mol1 = 0; mol1 < numberOfMoleculesInCell_i; ++mol1)
            {
                auto *molecule1 = cell1.getMolecule(mol1);

                for (const auto atom1 : cell1.getAtomIndices(mol1))
                {
                    for (size_t mol2 = 0; mol2 < numberOfMoleculesInCell_j; ++mol2)
                    {
                        auto *molecule2 = cell2->getMolecule(mol2);

                        if (molecule1 == molecule2)
                            continue;

                        for (const auto atom2 : cell2->getAtomIndices(mol2))
                        {
                            const auto [coulombEnergy, nonCoulombEnergy] =
                                calculateSingleInteraction(*box, *molecule1, *molecule2, atom1, atom2);

                            totalCoulombEnergy    += coulombEnergy;
                            totalNonCoulombEnergy += nonCoulombEnergy;
                        }
                    }
                }
            }
        }
    }

    physicalData.setCoulombEnergy(totalCoulombEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);
}