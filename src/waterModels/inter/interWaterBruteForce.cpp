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

#include "interWater.hpp"   // for InterWater
#include "physicalData.hpp"
#include "potential.hpp"   // for ChargeTag

namespace waterModel
{

    /**
     * @brief Evaluate intermolecular water interactions by brute force.
     *
     * @details Iterates over all active water-molecule pairs, accumulates
     * Coulomb and non-Coulomb contributions, and adds forces directly to the
     * atoms.
     */
    void InterWaterStrategyBruteForce::calculate(
        const InterWaterState                        &state,
        molsys::SimulationBox                        &simulationBox,
        physicalData::PhysicalData                   &physicalData,
        const std::shared_ptr<pot::CoulombPotential> &coulPot,
        const molsys::CellList & /*cellList*/
    )
    {
        const auto rCut = pot::CoulombPotential::getCoulombRadiusCutOff();
        const auto rCutSquared = rCut * rCut;

        auto totalCoulombEnergy    = 0.0;
        auto totalNonCoulombEnergy = 0.0;

        size_t idxI = 0;
        for (auto &water1 : simulationBox.getWaterTypeMolecules())
        {
            size_t idxJ = 0;
            for (auto &water2 : simulationBox.getWaterTypeMolecules())
            {
                // avoid double counting and self interaction
                if (idxJ >= idxI)
                    break;

                auto &oxygen1   = water1.getAtom(AtomIndex{0});
                auto &oxygen2   = water2.getAtom(AtomIndex{0});
                auto &hydrogen1 = water1.getAtom(AtomIndex{1});
                auto &hydrogen2 = water1.getAtom(AtomIndex{2});
                auto &hydrogen3 = water2.getAtom(AtomIndex{1});
                auto &hydrogen4 = water2.getAtom(AtomIndex{2});

                const auto singleInteraction =
                    [&](auto &atomA, auto &atomB, const auto &nonCoulPairPtr)
                {
                    if (nonCoulPairPtr)
                    {
                        calculateSingleInteraction<
                            pot::MMChargeTag,
                            pot::MMChargeTag>(
                            atomA,
                            atomB,
                            coulPot,
                            rCutSquared,
                            simulationBox,
                            *nonCoulPairPtr,
                            totalCoulombEnergy,
                            totalNonCoulombEnergy
                        );
                    }
                };

                // O-O interaction
                singleInteraction(oxygen1, oxygen2, state.nonCoulombPairOO);

                // O-H interactions
                singleInteraction(oxygen1, hydrogen3, state.nonCoulombPairOH);
                singleInteraction(oxygen1, hydrogen4, state.nonCoulombPairOH);
                singleInteraction(hydrogen1, oxygen2, state.nonCoulombPairOH);
                singleInteraction(hydrogen2, oxygen2, state.nonCoulombPairOH);

                // H-H interactions
                singleInteraction(hydrogen1, hydrogen3, state.nonCoulombPairHH);
                singleInteraction(hydrogen1, hydrogen4, state.nonCoulombPairHH);
                singleInteraction(hydrogen2, hydrogen3, state.nonCoulombPairHH);
                singleInteraction(hydrogen2, hydrogen4, state.nonCoulombPairHH);

                ++idxJ;
            }
            ++idxI;
        }

        physicalData.addCoulombEnergy(totalCoulombEnergy);
        physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
    }

    /**
     * @brief Compute core-to-outer Coulomb interactions by brute force.
     *
     * @param simulationBox Simulation box containing molecules.
     * @param physicalData Physical data to store energy results.
     * @param coulombPotential Coulomb potential evaluator.
     */
    void InterWaterStrategyBruteForce::calculateCoreToOuterForces(
        const InterWaterState & /*state*/,
        molsys::SimulationBox                        &simulationBox,
        physicalData::PhysicalData                   &physicalData,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        const molsys::CellList & /*cellList*/
    )
    {
        const auto rCut = pot::CoulombPotential::getCoulombRadiusCutOff();
        const auto rCutSquared = rCut * rCut;

        auto totalCoulombEnergy = 0.0;

        const auto waterTypeValue =
            simulationBox.getWaterType().value_or(MolType{0});

        for (auto &water1 :
             simulationBox.getMoleculesInsideZone(molsys::HybridZone::CORE))
        {
            if (water1.getMoltype() != waterTypeValue)
                continue;

            for (auto &water2 : simulationBox.getMMMolecules())
            {
                if (water2.getMoltype() != waterTypeValue)
                    continue;

                auto &oxygen1   = water1.getAtom(AtomIndex{0});
                auto &oxygen2   = water2.getAtom(AtomIndex{0});
                auto &hydrogen1 = water1.getAtom(AtomIndex{1});
                auto &hydrogen2 = water1.getAtom(AtomIndex{2});
                auto &hydrogen3 = water2.getAtom(AtomIndex{1});
                auto &hydrogen4 = water2.getAtom(AtomIndex{2});

                const auto singleCoulombInteraction =
                    [&](auto &atomA, auto &atomB)
                {
                    calculateSingleCoulombInteraction<
                        pot::QMChargeTag,
                        pot::MMChargeTag>(
                        atomA,
                        atomB,
                        coulombPotential,
                        rCutSquared,
                        simulationBox,
                        totalCoulombEnergy
                    );
                };

                // O-O interaction
                singleCoulombInteraction(oxygen1, oxygen2);

                // O-H interactions
                singleCoulombInteraction(oxygen1, hydrogen3);
                singleCoulombInteraction(oxygen1, hydrogen4);
                singleCoulombInteraction(hydrogen1, oxygen2);
                singleCoulombInteraction(hydrogen2, oxygen2);

                // H-H interactions
                singleCoulombInteraction(hydrogen1, hydrogen3);
                singleCoulombInteraction(hydrogen1, hydrogen4);
                singleCoulombInteraction(hydrogen2, hydrogen3);
                singleCoulombInteraction(hydrogen2, hydrogen4);
            }
        }

        physicalData.addCoulombEnergy(totalCoulombEnergy);
    }

    /**
     * @brief Compute layer-to-outer Coulomb and non-Coulomb interactions.
     *
     * @param state Inter-water parameters.
     * @param simulationBox Simulation box containing molecules.
     * @param physicalData Physical data to store energy results.
     * @param coulombPotential Coulomb potential evaluator.
     */
    void InterWaterStrategyBruteForce::calculateLayerToOuterForces(
        const InterWaterState                        &state,
        molsys::SimulationBox                        &simulationBox,
        physicalData::PhysicalData                   &physicalData,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        const molsys::CellList & /*cellList*/
    )
    {
        const auto rCut = pot::CoulombPotential::getCoulombRadiusCutOff();
        const auto rCutSquared = rCut * rCut;

        auto totalCoulombEnergy    = 0.0;
        auto totalNonCoulombEnergy = 0.0;

        const auto waterTypeValue =
            simulationBox.getWaterType().value_or(MolType{0});

        for (auto &water1 : simulationBox.getInactiveMolecules())
        {
            if (water1.getHybridZone() == molsys::HybridZone::CORE)
                continue;

            if (water1.getMoltype() != waterTypeValue)
                continue;

            for (auto &water2 : simulationBox.getMMMolecules())
            {
                if (water2.getMoltype() != waterTypeValue)
                    continue;

                const auto singleInteraction =
                    [&](auto &atomA, auto &atomB, const auto &nonCoulPairPtr)
                {
                    if (nonCoulPairPtr)
                    {
                        calculateSingleInteraction<
                            pot::QMChargeTag,
                            pot::MMChargeTag>(
                            atomA,
                            atomB,
                            coulombPotential,
                            rCutSquared,
                            simulationBox,
                            *nonCoulPairPtr,
                            totalCoulombEnergy,
                            totalNonCoulombEnergy
                        );
                    }
                };

                auto &oxygen1   = water1.getAtom(AtomIndex{0});
                auto &oxygen2   = water2.getAtom(AtomIndex{0});
                auto &hydrogen1 = water1.getAtom(AtomIndex{1});
                auto &hydrogen2 = water1.getAtom(AtomIndex{2});
                auto &hydrogen3 = water2.getAtom(AtomIndex{1});
                auto &hydrogen4 = water2.getAtom(AtomIndex{2});

                // O-O interaction
                singleInteraction(oxygen1, oxygen2, state.nonCoulombPairOO);

                // O-H interactions
                singleInteraction(oxygen1, hydrogen3, state.nonCoulombPairOH);
                singleInteraction(oxygen1, hydrogen4, state.nonCoulombPairOH);
                singleInteraction(hydrogen1, oxygen2, state.nonCoulombPairOH);
                singleInteraction(hydrogen2, oxygen2, state.nonCoulombPairOH);

                // H-H interactions
                singleInteraction(hydrogen1, hydrogen3, state.nonCoulombPairHH);
                singleInteraction(hydrogen1, hydrogen4, state.nonCoulombPairHH);
                singleInteraction(hydrogen2, hydrogen3, state.nonCoulombPairHH);
                singleInteraction(hydrogen2, hydrogen4, state.nonCoulombPairHH);
            }
        }

        physicalData.addCoulombEnergy(totalCoulombEnergy);
        physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
    }

    /**
     * @brief Compute outer-to-outer interactions by brute force.
     *
     * @param state Inter-water parameters.
     * @param simulationBox Simulation box containing molecules.
     * @param physicalData Physical data to store energy results.
     * @param coulombPotential Coulomb potential evaluator.
     * @param cellList Cell list structure (unused).
     */
    void InterWaterStrategyBruteForce::calculateOuterToOuterForces(
        const InterWaterState                        &state,
        molsys::SimulationBox                        &simulationBox,
        physicalData::PhysicalData                   &physicalData,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        const molsys::CellList                       &cellList
    )
    {
        calculate(
            state,
            simulationBox,
            physicalData,
            coulombPotential,
            cellList
        );
    }

    /**
     * @brief Compute smoothing-zone interactions against MM molecules.
     *
     * @param state Inter-water parameters.
     * @param simulationBox Simulation box containing molecules.
     * @param physicalData Physical data to store energy results.
     * @param coulombPotential Coulomb potential evaluator.
     */
    void InterWaterStrategyBruteForce::calculateHotspotSmoothingMMForces(
        const InterWaterState                        &state,
        molsys::SimulationBox                        &simulationBox,
        physicalData::PhysicalData                   &physicalData,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        const molsys::CellList & /*cellList*/
    )
    {
        const auto rCut = pot::CoulombPotential::getCoulombRadiusCutOff();
        const auto rCutSquared = rCut * rCut;

        auto totalCoulombEnergy    = 0.0;
        auto totalNonCoulombEnergy = 0.0;

        const auto waterTypeValue =
            simulationBox.getWaterType().value_or(MolType{0});

        for (auto &water1 : simulationBox.getMoleculesInsideZone(
                 molsys::HybridZone::SMOOTHING
             ))
        {
            if (water1.getMoltype() != waterTypeValue)
                continue;

            for (auto &water2 : simulationBox.getMoleculesOutsideZone(
                     molsys::HybridZone::SMOOTHING
                 ))
            {
                if (water2.getMoltype() != waterTypeValue)
                    continue;

                auto &oxygen1   = water1.getAtom(AtomIndex{0});
                auto &oxygen2   = water2.getAtom(AtomIndex{0});
                auto &hydrogen1 = water1.getAtom(AtomIndex{1});
                auto &hydrogen2 = water1.getAtom(AtomIndex{2});
                auto &hydrogen3 = water2.getAtom(AtomIndex{1});
                auto &hydrogen4 = water2.getAtom(AtomIndex{2});

                const auto singleInteraction =
                    [&](auto &atomA, auto &atomB, const auto &nonCoulPairPtr)
                {
                    if (nonCoulPairPtr)
                    {
                        calculateSingleInteraction<
                            pot::MMChargeTag,
                            pot::QMChargeTag>(
                            atomA,
                            atomB,
                            coulombPotential,
                            rCutSquared,
                            simulationBox,
                            *nonCoulPairPtr,
                            totalCoulombEnergy,
                            totalNonCoulombEnergy
                        );
                    }
                };

                // O-O interaction
                singleInteraction(oxygen1, oxygen2, state.nonCoulombPairOO);

                // O-H interactions
                singleInteraction(oxygen1, hydrogen3, state.nonCoulombPairOH);
                singleInteraction(oxygen1, hydrogen4, state.nonCoulombPairOH);
                singleInteraction(hydrogen1, oxygen2, state.nonCoulombPairOH);
                singleInteraction(hydrogen2, oxygen2, state.nonCoulombPairOH);

                // H-H interactions
                singleInteraction(hydrogen1, hydrogen3, state.nonCoulombPairHH);
                singleInteraction(hydrogen1, hydrogen4, state.nonCoulombPairHH);
                singleInteraction(hydrogen2, hydrogen3, state.nonCoulombPairHH);
                singleInteraction(hydrogen2, hydrogen4, state.nonCoulombPairHH);
            }
        }

        size_t idxI = 0;
        for (auto &water1 : simulationBox.getMoleculesInsideZone(
                 molsys::HybridZone::SMOOTHING
             ))
        {
            if (water1.getMoltype() != waterTypeValue)
            {
                ++idxI;
                continue;
            }

            size_t idxJ = 0;
            for (auto &water2 : simulationBox.getMoleculesInsideZone(
                     molsys::HybridZone::SMOOTHING
                 ))
            {
                if (water2.getMoltype() != waterTypeValue)
                {
                    ++idxJ;
                    continue;
                }

                if (idxI == idxJ)
                {
                    ++idxJ;
                    continue;
                }

                const auto singleInteractionOneWay =
                    [&](auto &atomA, auto &atomB, const auto &nonCoulPairPtr)
                {
                    if (nonCoulPairPtr)
                    {
                        calculateSingleInteractionOneWay<
                            pot::MMChargeTag,
                            pot::QMChargeTag>(
                            atomA,
                            atomB,
                            coulombPotential,
                            rCutSquared,
                            simulationBox,
                            *nonCoulPairPtr,
                            totalCoulombEnergy,
                            totalNonCoulombEnergy
                        );
                    }
                };

                auto &oxygen1   = water1.getAtom(AtomIndex{0});
                auto &oxygen2   = water2.getAtom(AtomIndex{0});
                auto &hydrogen1 = water1.getAtom(AtomIndex{1});
                auto &hydrogen2 = water1.getAtom(AtomIndex{2});
                auto &hydrogen3 = water2.getAtom(AtomIndex{1});
                auto &hydrogen4 = water2.getAtom(AtomIndex{2});

                // O-O interaction
                singleInteractionOneWay(
                    oxygen1,
                    oxygen2,
                    state.nonCoulombPairOO
                );

                // O-H interactions
                singleInteractionOneWay(
                    oxygen1,
                    hydrogen3,
                    state.nonCoulombPairOH
                );
                singleInteractionOneWay(
                    oxygen1,
                    hydrogen4,
                    state.nonCoulombPairOH
                );
                singleInteractionOneWay(
                    hydrogen1,
                    oxygen2,
                    state.nonCoulombPairOH
                );
                singleInteractionOneWay(
                    hydrogen2,
                    oxygen2,
                    state.nonCoulombPairOH
                );

                // H-H interactions
                singleInteractionOneWay(
                    hydrogen1,
                    hydrogen3,
                    state.nonCoulombPairHH
                );
                singleInteractionOneWay(
                    hydrogen1,
                    hydrogen4,
                    state.nonCoulombPairHH
                );
                singleInteractionOneWay(
                    hydrogen2,
                    hydrogen3,
                    state.nonCoulombPairHH
                );
                singleInteractionOneWay(
                    hydrogen2,
                    hydrogen4,
                    state.nonCoulombPairHH
                );

                ++idxJ;
            }
            ++idxI;
        }

        physicalData.addCoulombEnergy(totalCoulombEnergy);
        physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
    }

}   // namespace waterModel
