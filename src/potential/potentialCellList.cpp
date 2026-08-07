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

#include "potentialCellList.hpp"   // for PotentialCellList

#include <cstddef>   // for size_t
#include <vector>    // for vector

#include "cell.hpp"            // for Cell, simulationBox
#include "celllist.hpp"        // for CellList
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox

using namespace potential;
using namespace simulationBox;
using namespace physicalData;

/**
 * @brief Destroy the Potential Cell List:: Potential Cell List object
 *
 */
PotentialCellList::~PotentialCellList() = default;

/**
 * @brief calculates forces, coulombic and non-coulombic energy for cell list
 * routine
 *
 * @details first loops over all possible combinations of molecules within the
 * same cell, then over all possible molecule combinations between adjacent
 * cells. For the second loop over different cells, it is necessary to check if
 * the two molecules are the same to avoid double counting. Due to the cutoff
 * criterion which is based on atoms a molecule can be found in more than only
 * one cell.
 *
 * @param simBox
 * @param physicalData
 * @param cellList
 */
inline void PotentialCellList::calculateForces(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    CellList      &cellList
)
{
    startTimingsSection("InterNonBonded");

    const auto box = simBox.getBoxPtr();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    for (const auto &cell_i : cellList.getCells())
    {
        const auto nMols = cell_i.getNumberOfMolecules();

        for (size_t mol_i = 0; mol_i < nMols; ++mol_i)
        {
            auto *molecule_i = cell_i.getMolecule(mol_i);

            for (size_t mol_j = 0; mol_j < mol_i; ++mol_j)
            {
                auto *molecule_j = cell_i.getMolecule(mol_j);

                for (const size_t atom_i : cell_i.getAtomIndices(mol_i))
                {
                    for (const size_t atom_j : cell_i.getAtomIndices(mol_j))
                    {
                        const auto [coulombEnergy, nonCoulombEnergy] =
                            calculateSingleInteraction(
                                *box,
                                *molecule_i,
                                *molecule_j,
                                atom_i,
                                atom_j
                            );

                        totalCoulombEnergy    += coulombEnergy;
                        totalNonCoulombEnergy += nonCoulombEnergy;
                    }
                }
            }
        }
    }

    for (const auto &cell_i : cellList.getCells())
    {
        const auto nMolsInCell_i = cell_i.getNumberOfMolecules();

        for (const auto *cell_j : cell_i.getNeighbourCells())
        {
            const auto nMolsInCell_j = cell_j->getNumberOfMolecules();

            for (size_t mol_i = 0; mol_i < nMolsInCell_i; ++mol_i)
            {
                auto *molecule_i = cell_i.getMolecule(mol_i);

                for (const auto atom_i : cell_i.getAtomIndices(mol_i))
                {
                    for (size_t mol_j = 0; mol_j < nMolsInCell_j; ++mol_j)
                    {
                        auto *molecule_j = cell_j->getMolecule(mol_j);

                        if (molecule_i == molecule_j)
                            continue;

                        for (const auto atom_j : cell_j->getAtomIndices(mol_j))
                        {
                            const auto [coulombEnergy, nonCoulombEnergy] =
                                calculateSingleInteraction(
                                    *box,
                                    *molecule_i,
                                    *molecule_j,
                                    atom_i,
                                    atom_j
                                );

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

    stopTimingsSection("InterNonBonded");
}

/**
 * @brief clone the potential
 *
 * @return std::shared_ptr<PotentialCellList>
 */
std::shared_ptr<Potential> PotentialCellList::clone() const
{
    return std::make_shared<PotentialCellList>(*this);
}