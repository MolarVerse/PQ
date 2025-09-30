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
#include "molecule.hpp"        // for Molecule
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox

using namespace potential;
using namespace simulationBox;
using namespace physicalData;

using enum ChargeType;
using enum simulationBox::HybridZone;

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
void PotentialCellList::calculateForces(
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
        size_t i = 0;
        for (auto &mol1 : cell_i.getMMMolecules())
        {
            size_t j = 0;
            for (auto &mol2 : cell_i.getMMMolecules())
            {
                // avoid double counting and self interaction
                if (j >= i)
                    break;

                for (auto &atom1 : mol1->getAtoms())
                    for (auto &atom2 : mol2->getAtoms())
                    {
                        const auto [coulombEnergy, nonCoulombEnergy] =
                            calculateSingleInteraction(
                                *box,
                                *mol1,
                                *mol2,
                                *atom1,
                                *atom2,
                                MM_CHARGE,
                                MM_CHARGE
                            );

                        totalCoulombEnergy    += coulombEnergy;
                        totalNonCoulombEnergy += nonCoulombEnergy;
                    }
                ++j;
            }
            ++i;
        }
    }

    for (const auto &cell1 : cellList.getCells())
        for (const auto *cell2 : cell1.getNeighbourCells())
            for (auto &mol1 : cell1.getMMMolecules())
                for (auto &mol2 : cell2->getMMMolecules())
                {
                    // avoid self interaction
                    if (mol1 == mol2)
                        continue;

                    for (auto &atom1 : mol1->getAtoms())
                        for (auto &atom2 : mol2->getAtoms())
                        {
                            const auto [coulombEnergy, nonCoulombEnergy] =
                                calculateSingleInteraction(
                                    *box,
                                    *mol1,
                                    *mol2,
                                    *atom1,
                                    *atom2,
                                    MM_CHARGE,
                                    MM_CHARGE
                                );

                            totalCoulombEnergy    += coulombEnergy;
                            totalNonCoulombEnergy += nonCoulombEnergy;
                        }
                }

    physicalData.setCoulombEnergy(totalCoulombEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);

    stopTimingsSection("InterNonBonded");
}

/**
 * @brief calculates Coulomb forces between core zone molecules and all MM
 * molecules using cell list optimization
 *
 * @details loops over all cells and calculates interactions between core zone
 * molecules and MM molecules within the same cell, then between core zone
 * molecules in one cell and MM molecules in neighboring cells. Uses cell list
 * structure for efficient neighbor searching.
 *
 * @param simBox simulation box containing molecules
 * @param physicalData physical data to store energy results
 * @param cellList cell list structure for efficient neighbor searching
 */
void PotentialCellList::calculateCoreToOuterForces(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    CellList      &cellList
)
{
    startTimingsSection("InterNonBondedCoreToOuter");

    const auto box = simBox.getBoxPtr();

    double totalCoulombEnergy = 0.0;

    for (const auto &cell_i : cellList.getCells())
        for (auto &mol1 : cell_i.getMoleculesInsideZone(CORE))
            for (auto &mol2 : cell_i.getMMMolecules())
                for (auto &atom1 : mol1->getAtoms())
                    for (auto &atom2 : mol2->getAtoms())
                        totalCoulombEnergy += calculateSingleCoulombInteraction(
                            *box,
                            *atom1,
                            *atom2,
                            QM_CHARGE,
                            MM_CHARGE
                        );

    for (const auto &cell1 : cellList.getCells())
        for (const auto *cell2 : cell1.getNeighbourCells())
            for (auto &mol1 : cell1.getMoleculesInsideZone(CORE))
                for (auto &mol2 : cell2->getMMMolecules())
                    for (auto &atom1 : mol1->getAtoms())
                        for (auto &atom2 : mol2->getAtoms())
                            totalCoulombEnergy +=
                                calculateSingleCoulombInteraction(
                                    *box,
                                    *atom1,
                                    *atom2,
                                    QM_CHARGE,
                                    MM_CHARGE
                                );

    physicalData.addCoulombEnergy(totalCoulombEnergy);

    stopTimingsSection("InterNonBondedCoreToOuter");
}

/**
 * @brief calculates forces between layer and outer molecules using cell list
 * optimization
 *
 * @details loops over all cells and calculates interactions between MM
 * molecules and inactive molecules within the same cell, then between MM
 * molecules in one cell and inactive molecules in neighboring cells. Skips
 * interactions with core zone molecules. Uses cell list structure for efficient
 * neighbor searching.
 *
 * @param simBox simulation box containing molecules
 * @param physicalData physical data to store energy results
 * @param cellList cell list structure for efficient neighbor searching
 */
void PotentialCellList::calculateLayerToOuterForces(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    CellList      &cellList
)
{
    startTimingsSection("InterNonBondedLayerToOuter");

    const auto box = simBox.getBoxPtr();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    for (const auto &cell_i : cellList.getCells())
        for (auto &mol1 : cell_i.getInactiveMolecules())
        {
            if (mol1->getHybridZone() == CORE)
                continue;

            for (auto &mol2 : cell_i.getMMMolecules())
                for (auto &atom1 : mol1->getAtoms())
                    for (auto &atom2 : mol2->getAtoms())
                    {
                        const auto [coulombEnergy, nonCoulombEnergy] =
                            calculateSingleInteraction(
                                *box,
                                *mol1,
                                *mol2,
                                *atom1,
                                *atom2,
                                QM_CHARGE,
                                MM_CHARGE
                            );

                        totalCoulombEnergy    += coulombEnergy;
                        totalNonCoulombEnergy += nonCoulombEnergy;
                    }
        }

    for (const auto &cell1 : cellList.getCells())
        for (const auto *cell2 : cell1.getNeighbourCells())
            for (auto &mol1 : cell1.getMMMolecules())
                for (auto &mol2 : cell2->getInactiveMolecules())
                {
                    if (mol2->getHybridZone() == CORE)
                        continue;

                    for (auto &atom1 : mol1->getAtoms())
                        for (auto &atom2 : mol2->getAtoms())
                        {
                            const auto [coulombEnergy, nonCoulombEnergy] =
                                calculateSingleInteraction(
                                    *box,
                                    *mol1,
                                    *mol2,
                                    *atom1,
                                    *atom2,
                                    QM_CHARGE,
                                    MM_CHARGE
                                );

                            totalCoulombEnergy    += coulombEnergy;
                            totalNonCoulombEnergy += nonCoulombEnergy;
                        }
                }

    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);

    stopTimingsSection("InterNonBondedLayerToOuter");
}

void PotentialCellList::calculateHotspotSmoothingMMForces(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    CellList      &cellList
)
{
    startTimingsSection("InterNonBondedSmoothingMM");

    const auto box = simBox.getBoxPtr();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    for (const auto &cell_i : cellList.getCells())
        for (auto &mol1 : cell_i.getMoleculesInsideZone(SMOOTHING))
            for (auto &mol2 : cell_i.getMoleculesOutsideZone(SMOOTHING))
                for (auto &atom1 : mol1->getAtoms())
                    for (auto &atom2 : mol2->getAtoms())
                    {
                        const auto [coulombEnergy, nonCoulombEnergy] =
                            calculateSingleInteraction(
                                *box,
                                *mol1,
                                *mol2,
                                *atom1,
                                *atom2,
                                MM_CHARGE,
                                QM_CHARGE
                            );

                        totalCoulombEnergy    += coulombEnergy;
                        totalNonCoulombEnergy += nonCoulombEnergy;
                    }

    for (const auto &cell1 : cellList.getCells())
        for (const auto *cell2 : cell1.getNeighbourCells())
            for (auto &mol1 : cell1.getMoleculesInsideZone(SMOOTHING))
                for (auto &mol2 : cell2->getMoleculesOutsideZone(SMOOTHING))
                    for (auto &atom1 : mol1->getAtoms())
                        for (auto &atom2 : mol2->getAtoms())
                        {
                            const auto [coulombEnergy, nonCoulombEnergy] =
                                calculateSingleInteraction(
                                    *box,
                                    *mol1,
                                    *mol2,
                                    *atom1,
                                    *atom2,
                                    MM_CHARGE,
                                    QM_CHARGE
                                );

                            totalCoulombEnergy    += coulombEnergy;
                            totalNonCoulombEnergy += nonCoulombEnergy;
                        }

    for (const auto &cell_i : cellList.getCells())
    {
        size_t i = 0;
        for (auto &mol1 : cell_i.getMoleculesInsideZone(SMOOTHING))
        {
            size_t j = 0;
            for (auto &mol2 : cell_i.getMoleculesInsideZone(SMOOTHING))
            {
                if (i == j)
                {
                    ++j;
                    continue;
                }
                for (auto &atom1 : mol1->getAtoms())
                    for (auto &atom2 : mol2->getAtoms())
                    {
                        const auto [coulombEnergy, nonCoulombEnergy] =
                            calculateSingleInteractionOneWay(
                                *box,
                                *mol1,
                                *mol2,
                                *atom1,
                                *atom2,
                                MM_CHARGE,
                                QM_CHARGE
                            );

                        totalCoulombEnergy    += coulombEnergy;
                        totalNonCoulombEnergy += nonCoulombEnergy;
                    }
                ++j;
            }
            ++i;
        }
    }

    for (const auto &cell1 : cellList.getCells())
        for (const auto *cell2 : cell1.getNeighbourCells())
        {
            size_t i = 0;
            for (auto &mol1 : cell1.getMoleculesInsideZone(SMOOTHING))
            {
                size_t j = 0;
                for (auto &mol2 : cell2->getMoleculesInsideZone(SMOOTHING))
                {
                    if (i == j)
                    {
                        ++j;
                        continue;
                    }
                    for (auto &atom1 : mol1->getAtoms())
                        for (auto &atom2 : mol2->getAtoms())
                        {
                            const auto [coulombEnergy, nonCoulombEnergy] =
                                calculateSingleInteractionOneWay(
                                    *box,
                                    *mol1,
                                    *mol2,
                                    *atom1,
                                    *atom2,
                                    MM_CHARGE,
                                    QM_CHARGE
                                );

                            totalCoulombEnergy    += coulombEnergy;
                            totalNonCoulombEnergy += nonCoulombEnergy;
                        }
                    ++j;
                }
                ++i;
            }
        }

    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);

    stopTimingsSection("InterNonBondedSmoothingMM");
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