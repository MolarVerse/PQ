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

#include "potentialBruteForce.hpp"

#include <cstddef>

#include "globalTimer.hpp"
#include "molecule.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "waterModelSettings.hpp"

namespace pot
{

    /**
     * @brief Destroy the Potential Brute Force:: Potential Brute Force object
     *
     */
    PotentialBruteForce::~PotentialBruteForce() = default;

    /**
     * @brief calculates forces, coulombic and non-coulombic energy for brute
     * force routine
     *
     * @param simulationBox
     * @param physicalData
     */
    void PotentialBruteForce::calculateForces(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const molsys::CellList & /*cellList*/
    )
    {
        auto _ = scopedTimer(TimerId::Potential, "InterNonBonded");

        const auto box = simulationBox.getBoxPtr();
        const auto waterTypeValue =
            simulationBox.getWaterType().value_or(MolType{0});
        const auto isWaterInterModelSet =
            settings::WaterModelSettings::isInterWaterModelSet();

        double totalCoulombEnergy    = 0.0;
        double totalNonCoulombEnergy = 0.0;

        size_t idxI = 0;
        for (auto &mol1 : simulationBox.getMMMolecules())
        {
            const auto isMol1Water = mol1.getMoltype() == waterTypeValue;

            size_t idxJ = 0;
            for (auto &mol2 : simulationBox.getMMMolecules())
            {
                // avoid double counting and self interaction
                if (idxJ >= idxI)
                    break;

                if (isWaterInterModelSet && isMol1Water &&
                    mol2.getMoltype() == waterTypeValue)
                {
                    ++idxJ;
                    continue;
                }

                for (auto &atom1 : mol1.getAtoms())
                {
                    for (auto &atom2 : mol2.getAtoms())
                    {
                        const auto [coulombEnergy, nonCoulombEnergy] =
                            calculateSingleInteraction<
                                MMChargeTag,
                                MMChargeTag>(*box, mol1, mol2, *atom1, *atom2);

                        totalCoulombEnergy    += coulombEnergy;
                        totalNonCoulombEnergy += nonCoulombEnergy;
                    }
                }
                ++idxJ;
            }
            ++idxI;
        }

        physicalData.addCoulombEnergy(totalCoulombEnergy);
        physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
    }

    /**
     * @brief calculates Coulomb forces between core zone molecules and all
     * MM molecules
     *
     * @param simulationBox simulation box containing molecules
     * @param physicalData physical data to store energy results
     */
    void PotentialBruteForce::calculateCoreToOuterForces(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const molsys::CellList & /*cellList*/
    )
    {
        auto _ = scopedTimer(TimerId::Potential, "InterNonBondedCoreToOuter");

        const auto box = simulationBox.getBoxPtr();

        double totalCoulombEnergy = 0.0;

        const auto waterTypeValue =
            simulationBox.getWaterType().value_or(MolType{0});
        const auto isWaterInterModelSet =
            settings::WaterModelSettings::isInterWaterModelSet();

        for (auto &mol1 :
             simulationBox.getMoleculesInsideZone(molsys::HybridZone::CORE))
        {
            const auto isMol1Water = mol1.getMoltype() == waterTypeValue;

            for (auto &mol2 : simulationBox.getMMMolecules())
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
     * @param simulationBox simulation box containing molecules
     * @param physicalData physical data to store energy results
     */
    void PotentialBruteForce::calculateLayerToOuterForces(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const molsys::CellList & /*cellList*/
    )
    {
        auto _ = scopedTimer(TimerId::Potential, "InterNonBondedLayerToOuter");

        const auto box = simulationBox.getBoxPtr();
        const auto waterTypeValue =
            simulationBox.getWaterType().value_or(MolType{0});
        const auto isWaterInterModelSet =
            settings::WaterModelSettings::isInterWaterModelSet();

        double totalCoulombEnergy    = 0.0;
        double totalNonCoulombEnergy = 0.0;

        for (auto &mol1 : simulationBox.getInactiveMolecules())
        {
            if (mol1.getHybridZone() == molsys::HybridZone::CORE)
                continue;

            const auto isMol1Water = mol1.getMoltype() == waterTypeValue;

            for (auto &mol2 : simulationBox.getMMMolecules())
            {
                if (isWaterInterModelSet && isMol1Water &&
                    mol2.getMoltype() == waterTypeValue)
                    continue;

                for (auto &atom1 : mol1.getAtoms())
                {
                    for (auto &atom2 : mol2.getAtoms())
                    {
                        const auto [coulombEnergy, nonCoulombEnergy] =
                            calculateSingleInteraction<
                                QMChargeTag,
                                MMChargeTag>(*box, mol1, mol2, *atom1, *atom2);

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
     * @param simulationBox simulation box containing molecules
     * @param physicalData physical data to store energy results
     * @param cellList cell list (unused in brute force approach)
     */
    void PotentialBruteForce::calculateOuterToOuterForces(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const molsys::CellList     &cellList
    )
    {
        calculateForces(simulationBox, physicalData, cellList);
    }

    /**
     * @brief calculates forces between smoothing-zone molecules and all others
     *
     * @param simulationBox simulation box containing molecules
     * @param physicalData physical data to store energy results
     */
    void PotentialBruteForce::calculateHotspotSmoothingMMForces(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const molsys::CellList & /*cellList*/
    )
    {
        auto _ = scopedTimer(TimerId::Potential, "InterNonBondedSmoothingMM");

        const auto box = simulationBox.getBoxPtr();
        const auto waterTypeValue =
            simulationBox.getWaterType().value_or(MolType{0});
        const auto isWaterInterModelSet =
            settings::WaterModelSettings::isInterWaterModelSet();

        double totalCoulombEnergy    = 0.0;
        double totalNonCoulombEnergy = 0.0;

        for (auto &mol1 : simulationBox.getMoleculesInsideZone(
                 molsys::HybridZone::SMOOTHING
             ))
        {
            const auto isMol1Water = mol1.getMoltype() == waterTypeValue;

            for (auto &mol2 : simulationBox.getMoleculesOutsideZone(
                     molsys::HybridZone::SMOOTHING
                 ))
            {
                if (isWaterInterModelSet && isMol1Water &&
                    mol2.getMoltype() == waterTypeValue)
                    continue;

                const auto isMol2Core =
                    mol2.getHybridZone() == molsys::HybridZone::CORE;

                // SMOOTHING-CORE interaction: evaluate Coulomb term only
                if (isMol2Core)
                {
                    for (auto &atom1 : mol1.getAtoms())
                    {
                        for (auto &atom2 : mol2.getAtoms())
                        {
                            totalCoulombEnergy +=
                                calculateSingleCoulombInteraction<
                                    MMChargeTag,
                                    QMChargeTag>(*box, *atom1, *atom2);
                        }
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
                                    QMChargeTag>(
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
        }

        size_t idxI = 0;
        for (auto &mol1 : simulationBox.getMoleculesInsideZone(
                 molsys::HybridZone::SMOOTHING
             ))
        {
            const auto isMol1Water = mol1.getMoltype() == waterTypeValue;

            size_t idxJ = 0;
            for (auto &mol2 : simulationBox.getMoleculesInsideZone(
                     molsys::HybridZone::SMOOTHING
                 ))
            {
                if (idxI == idxJ)
                {
                    ++idxJ;
                    continue;
                }

                if (isWaterInterModelSet && isMol1Water &&
                    mol2.getMoltype() == waterTypeValue)
                {
                    ++idxJ;
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
                ++idxJ;
            }
            ++idxI;
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

}   // namespace pot
