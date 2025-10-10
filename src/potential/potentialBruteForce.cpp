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

#include "potentialBruteForce.hpp"   // for PotentialBruteForce

#include <cstddef>   // for size_t

#include "molecule.hpp"        // for Molecule
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox

using namespace potential;
using namespace simulationBox;
using namespace physicalData;

using enum simulationBox::HybridZone;

/**
 * @brief Destroy the Potential Brute Force:: Potential Brute Force object
 *
 */
PotentialBruteForce::~PotentialBruteForce() = default;

/**
 * @brief calculates forces, coulombic and non-coulombic energy for brute force
 * routine
 *
 * @param simBox
 * @param physicalData
 */
void PotentialBruteForce::calculateForces(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    CellList &
)
{
    startTimingsSection("InterNonBonded");

    const auto box = simBox.getBoxPtr();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    // inter molecular forces
    size_t i = 0;
    for (auto &mol1 : simBox.getMMMolecules())
    {
        size_t j = 0;
        for (auto &mol2 : simBox.getMMMolecules())
        {
            // avoid double counting and self interaction
            if (j >= i)
                break;

            for (auto &atom1 : mol1.getAtoms())
                for (auto &atom2 : mol2.getAtoms())
                {
                    const auto [coulombEnergy, nonCoulombEnergy] =
                        calculateSingleInteraction<MMChargeTag, MMChargeTag>(
                            *box,
                            mol1,
                            mol2,
                            *atom1,
                            *atom2
                        );

                    totalCoulombEnergy    += coulombEnergy;
                    totalNonCoulombEnergy += nonCoulombEnergy;
                }
            ++j;
        }
        ++i;
    }
    physicalData.setCoulombEnergy(totalCoulombEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);

    stopTimingsSection("InterNonBonded");
}

/**
 * @brief calculates Coulomb forces between core zone molecules and all
 * MM molecules
 *
 * @param simBox simulation box containing molecules
 * @param physicalData physical data to store energy results
 * @param unused CellList parameter (not used in brute force approach)
 */
void PotentialBruteForce::calculateCoreToOuterForces(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    CellList &
)
{
    startTimingsSection("InterNonBondedCoreToOuter");

    const auto box = simBox.getBoxPtr();

    double totalCoulombEnergy = 0.0;

    // inter molecular Coulomb forces
    for (auto &mol1 : simBox.getMoleculesInsideZone(CORE))
        for (auto &mol2 : simBox.getMMMolecules())
            for (auto &atom1 : mol1.getAtoms())
                for (auto &atom2 : mol2.getAtoms())
                    totalCoulombEnergy += calculateSingleCoulombInteraction<
                        QMChargeTag,
                        MMChargeTag>(*box, *atom1, *atom2);

    physicalData.addCoulombEnergy(totalCoulombEnergy);

    stopTimingsSection("InterNonBondedCoreToOuter");
}

/**
 * @brief calculates forces between layer and outer molecules
 *
 * @param simBox simulation box containing molecules
 * @param physicalData physical data to store energy results
 * @param unused CellList parameter (not used in brute force approach)
 */
void PotentialBruteForce::calculateLayerToOuterForces(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    CellList &
)
{
    startTimingsSection("InterNonBondedLayerToOuter");

    const auto box = simBox.getBoxPtr();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    // inter molecular forces
    for (auto &mol1 : simBox.getInactiveMolecules())
    {
        if (mol1.getHybridZone() == CORE)
            continue;

        for (auto &mol2 : simBox.getMMMolecules())
            for (auto &atom1 : mol1.getAtoms())
                for (auto &atom2 : mol2.getAtoms())
                {
                    const auto [coulombEnergy, nonCoulombEnergy] =
                        calculateSingleInteraction<QMChargeTag, MMChargeTag>(
                            *box,
                            mol1,
                            mol2,
                            *atom1,
                            *atom2
                        );

                    totalCoulombEnergy    += coulombEnergy;
                    totalNonCoulombEnergy += nonCoulombEnergy;
                }
    }
    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);

    stopTimingsSection("InterNonBondedLayerToOuter");
}

void PotentialBruteForce::calculateHotspotSmoothingMMForces(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    CellList &
)
{
    startTimingsSection("InterNonBondedSmoothingMM");

    const auto box = simBox.getBoxPtr();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    // inter molecular forces
    for (auto &mol1 : simBox.getMoleculesInsideZone(SMOOTHING))
        for (auto &mol2 : simBox.getMoleculesOutsideZone(SMOOTHING))
            for (auto &atom1 : mol1.getAtoms())
                for (auto &atom2 : mol2.getAtoms())
                {
                    const auto [coulombEnergy, nonCoulombEnergy] =
                        calculateSingleInteraction<MMChargeTag, QMChargeTag>(
                            *box,
                            mol1,
                            mol2,
                            *atom1,
                            *atom2
                        );

                    totalCoulombEnergy    += coulombEnergy;
                    totalNonCoulombEnergy += nonCoulombEnergy;
                }

    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);

    size_t i = 0;
    for (auto &mol1 : simBox.getMoleculesInsideZone(SMOOTHING))
    {
        size_t j = 0;
        for (auto &mol2 : simBox.getMoleculesInsideZone(SMOOTHING))
        {
            if (i == j)
            {
                ++j;
                continue;
            }

            for (auto &atom1 : mol1.getAtoms())
                for (auto &atom2 : mol2.getAtoms())
                {
                    const auto [coulombEnergy, nonCoulombEnergy] =
                        calculateSingleInteractionOneWay<
                            MMChargeTag,
                            QMChargeTag>(*box, mol1, mol2, *atom1, *atom2);

                    totalCoulombEnergy    += coulombEnergy;
                    totalNonCoulombEnergy += nonCoulombEnergy;
                }
            ++j;
        }
        ++i;
    }

    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);

    stopTimingsSection("InterNonBondedSmoothingMM");
}

/**
 * @brief clone the potential
 *
 * @return std::shared_ptr<PotentialBruteForce>
 */
std::shared_ptr<Potential> PotentialBruteForce::clone() const
{
    return std::make_shared<PotentialBruteForce>(*this);
}