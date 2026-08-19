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

#include "molecule.hpp"             // for Molecule
#include "physicalData.hpp"         // for PhysicalData
#include "simulationBox.hpp"        // for SimulationBox
#include "waterModelSettings.hpp"   // for WaterModelSettings

using namespace physicalData;
using namespace potential;
using namespace settings;
using namespace simulationBox;

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
    CellList & /*cellList*/
)
{
    auto _ = scoped("InterNonBonded");

    const auto box            = simBox.getBoxPtr();
    const auto waterTypeValue = simBox.getWaterType().value_or(size_t{0});
    const auto isWaterInterModelSet =
        WaterModelSettings::isInterWaterModelSet();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    size_t i = 0;
    for (auto &mol1 : simBox.getMMMolecules())
    {
        const auto isMol1Water = mol1.getMoltype() == waterTypeValue;

        size_t j = 0;
        for (auto &mol2 : simBox.getMMMolecules())
        {
            // avoid double counting and self interaction
            if (j >= i)
                break;

            if (isWaterInterModelSet && isMol1Water &&
                mol2.getMoltype() == waterTypeValue)
            {
                ++j;
                continue;
            }

            for (auto &atom1 : mol1.getAtoms())
            {
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
            }
            ++j;
        }
        ++i;
    }

    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
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
    CellList & /*cellList*/
)
{
    auto _ = scoped("InterNonBondedCoreToOuter");

    const auto box = simBox.getBoxPtr();

    double totalCoulombEnergy = 0.0;

    const auto waterTypeValue = simBox.getWaterType().value_or(size_t{0});
    const auto isWaterInterModelSet =
        WaterModelSettings::isInterWaterModelSet();

    for (auto &mol1 : simBox.getMoleculesInsideZone(CORE))
    {
        const auto isMol1Water = mol1.getMoltype() == waterTypeValue;

        for (auto &mol2 : simBox.getMMMolecules())
        {
            if (isWaterInterModelSet && isMol1Water &&
                mol2.getMoltype() == waterTypeValue)
                continue;

            for (auto &atom1 : mol1.getAtoms())
            {
                for (auto &atom2 : mol2.getAtoms())
                    totalCoulombEnergy += calculateSingleCoulombInteraction<
                        QMChargeTag,
                        MMChargeTag>(*box, *atom1, *atom2);
            }
        }
    }

    physicalData.addCoulombEnergy(totalCoulombEnergy);
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
    CellList & /*cellList*/
)
{
    auto _ = scoped("InterNonBondedLayerToOuter");

    const auto box            = simBox.getBoxPtr();
    const auto waterTypeValue = simBox.getWaterType().value_or(size_t{0});
    const auto isWaterInterModelSet =
        WaterModelSettings::isInterWaterModelSet();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    for (auto &mol1 : simBox.getInactiveMolecules())
    {
        if (mol1.getHybridZone() == CORE)
            continue;

        const auto isMol1Water = mol1.getMoltype() == waterTypeValue;

        for (auto &mol2 : simBox.getMMMolecules())
        {
            if (isWaterInterModelSet && isMol1Water &&
                mol2.getMoltype() == waterTypeValue)
                continue;

            for (auto &atom1 : mol1.getAtoms())
            {
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
        }
    }
    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
}

/**
 * @brief calculates forces between outer-zone molecules
 *
 * @param simBox simulation box containing molecules
 * @param physicalData physical data to store energy results
 * @param cellList cell list (unused in brute force approach)
 */
void PotentialBruteForce::calculateOuterToOuterForces(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    CellList      &cellList
)
{
    calculateForces(simBox, physicalData, cellList);
}

/**
 * @brief calculates forces between smoothing-zone molecules and all others
 *
 * @param simBox simulation box containing molecules
 * @param physicalData physical data to store energy results
 * @param unused CellList parameter (not used in brute force approach)
 */
void PotentialBruteForce::calculateHotspotSmoothingMMForces(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    CellList & /*cellList*/
)
{
    auto _ = scoped("InterNonBondedSmoothingMM");

    const auto box            = simBox.getBoxPtr();
    const auto waterTypeValue = simBox.getWaterType().value_or(size_t{0});
    const auto isWaterInterModelSet =
        WaterModelSettings::isInterWaterModelSet();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    for (auto &mol1 : simBox.getMoleculesInsideZone(SMOOTHING))
    {
        const auto isMol1Water = mol1.getMoltype() == waterTypeValue;

        for (auto &mol2 : simBox.getMoleculesOutsideZone(SMOOTHING))
        {
            if (isWaterInterModelSet && isMol1Water &&
                mol2.getMoltype() == waterTypeValue)
                continue;

            const auto isMol2Core = mol2.getHybridZone() == CORE;

            // SMOOTHING-CORE interaction: evaluate Coulomb term only
            if (isMol2Core)
            {
                for (auto &atom1 : mol1.getAtoms())
                {
                    for (auto &atom2 : mol2.getAtoms())
                        totalCoulombEnergy += calculateSingleCoulombInteraction<
                            MMChargeTag,
                            QMChargeTag>(*box, *atom1, *atom2);
                }
                // SMOOTHING-nonCORE: evaluate full interaction
            }
            else
            {
                for (auto &atom1 : mol1.getAtoms())
                {
                    for (auto &atom2 : mol2.getAtoms())
                    {
                        const auto [coulombEnergy, nonCoulombEnergy] =
                            calculateSingleInteraction<
                                MMChargeTag,
                                QMChargeTag>(*box, mol1, mol2, *atom1, *atom2);

                        totalCoulombEnergy    += coulombEnergy;
                        totalNonCoulombEnergy += nonCoulombEnergy;
                    }
                }
            }
        }
    }

    size_t i = 0;
    for (auto &mol1 : simBox.getMoleculesInsideZone(SMOOTHING))
    {
        const auto isMol1Water = mol1.getMoltype() == waterTypeValue;

        size_t j = 0;
        for (auto &mol2 : simBox.getMoleculesInsideZone(SMOOTHING))
        {
            if (i == j)
            {
                ++j;
                continue;
            }

            if (isWaterInterModelSet && isMol1Water &&
                mol2.getMoltype() == waterTypeValue)
            {
                ++j;
                continue;
            }

            for (auto &atom1 : mol1.getAtoms())
            {
                for (auto &atom2 : mol2.getAtoms())
                {
                    const auto [coulombEnergy, nonCoulombEnergy] =
                        calculateSingleInteractionOneWay<
                            MMChargeTag,
                            QMChargeTag>(*box, mol1, mol2, *atom1, *atom2);

                    totalCoulombEnergy    += coulombEnergy;
                    totalNonCoulombEnergy += nonCoulombEnergy;
                }
            }
            ++j;
        }
        ++i;
    }

    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
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
