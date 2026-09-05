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

using namespace pot;
using namespace pq;
using namespace waterModel;
using namespace physicalData;
using namespace molsys;

using enum molsys::HybridZone;

/**
 * @brief Evaluate intermolecular water interactions by brute force.
 *
 * @details Iterates over all active water-molecule pairs, accumulates Coulomb
 * and non-Coulomb contributions, and adds forces directly to the atoms.
 */
void InterWaterStrategyBruteForce::calculate(
    const InterWaterState                        &state,
    molsys::SimulationBox                        &simBox,
    physicalData::PhysicalData                   &physicalData,
    const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
    CellList & /*cellList*/
)
{
    const auto rCut        = pot::CoulombPotential::getCoulombRadiusCutOff();
    const auto rCutSquared = rCut * rCut;

    auto totalCoulombEnergy    = 0.0;
    auto totalNonCoulombEnergy = 0.0;

    size_t i = 0;
    for (auto &water1 : simBox.getWaterTypeMolecules())
    {
        size_t j = 0;
        for (auto &water2 : simBox.getWaterTypeMolecules())
        {
            // avoid double counting and self interaction
            if (j >= i)
                break;

            auto &oxygen1   = water1.getAtom(0);
            auto &oxygen2   = water2.getAtom(0);
            auto &hydrogen1 = water1.getAtom(1);
            auto &hydrogen2 = water1.getAtom(2);
            auto &hydrogen3 = water2.getAtom(1);
            auto &hydrogen4 = water2.getAtom(2);

            const auto singleInteraction =
                [&](Atom &atomA, Atom &atomB, const auto &nonCoulPairPtr)
            {
                if (nonCoulPairPtr)
                {
                    calculateSingleInteraction<MMChargeTag, MMChargeTag>(
                        atomA,
                        atomB,
                        coulombPotential,
                        rCutSquared,
                        simBox,
                        *nonCoulPairPtr,
                        totalCoulombEnergy,
                        totalNonCoulombEnergy
                    );
                }
            };

            // O-O interaction
            singleInteraction(oxygen1, oxygen2, state._nonCoulombPairOO);

            // O-H interactions
            singleInteraction(oxygen1, hydrogen3, state._nonCoulombPairOH);
            singleInteraction(oxygen1, hydrogen4, state._nonCoulombPairOH);
            singleInteraction(hydrogen1, oxygen2, state._nonCoulombPairOH);
            singleInteraction(hydrogen2, oxygen2, state._nonCoulombPairOH);

            // H-H interactions
            singleInteraction(hydrogen1, hydrogen3, state._nonCoulombPairHH);
            singleInteraction(hydrogen1, hydrogen4, state._nonCoulombPairHH);
            singleInteraction(hydrogen2, hydrogen3, state._nonCoulombPairHH);
            singleInteraction(hydrogen2, hydrogen4, state._nonCoulombPairHH);

            ++j;
        }
        ++i;
    }

    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
}

/**
 * @brief Compute core-to-outer Coulomb interactions by brute force.
 *
 * @param state Inter-water parameters.
 * @param simBox Simulation box containing molecules.
 * @param physicalData Physical data to store energy results.
 * @param coulombPotential Coulomb potential evaluator.
 */
void InterWaterStrategyBruteForce::calculateCoreToOuterForces(
    const InterWaterState & /*state*/,
    molsys::SimulationBox                        &simBox,
    PhysicalData                                 &physicalData,
    const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
    CellList & /*cellList*/
)
{
    const auto rCut        = pot::CoulombPotential::getCoulombRadiusCutOff();
    const auto rCutSquared = rCut * rCut;

    auto totalCoulombEnergy = 0.0;

    const auto waterTypeValue = simBox.getWaterType().value_or(size_t{0});

    for (auto &water1 : simBox.getMoleculesInsideZone(CORE))
    {
        if (water1.getMoltype() != waterTypeValue)
            continue;

        for (auto &water2 : simBox.getMMMolecules())
        {
            if (water2.getMoltype() != waterTypeValue)
                continue;

            auto &oxygen1   = water1.getAtom(0);
            auto &oxygen2   = water2.getAtom(0);
            auto &hydrogen1 = water1.getAtom(1);
            auto &hydrogen2 = water1.getAtom(2);
            auto &hydrogen3 = water2.getAtom(1);
            auto &hydrogen4 = water2.getAtom(2);

            const auto singleCoulombInteraction = [&](Atom &atomA, Atom &atomB)
            {
                calculateSingleCoulombInteraction<QMChargeTag, MMChargeTag>(
                    atomA,
                    atomB,
                    coulombPotential,
                    rCutSquared,
                    simBox,
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
 * @param simBox Simulation box containing molecules.
 * @param physicalData Physical data to store energy results.
 * @param coulombPotential Coulomb potential evaluator.
 */
void InterWaterStrategyBruteForce::calculateLayerToOuterForces(
    const InterWaterState                        &state,
    molsys::SimulationBox                        &simBox,
    PhysicalData                                 &physicalData,
    const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
    CellList & /*cellList*/
)
{
    const auto rCut        = pot::CoulombPotential::getCoulombRadiusCutOff();
    const auto rCutSquared = rCut * rCut;

    auto totalCoulombEnergy    = 0.0;
    auto totalNonCoulombEnergy = 0.0;

    const auto waterTypeValue = simBox.getWaterType().value_or(size_t{0});

    for (auto &water1 : simBox.getInactiveMolecules())
    {
        if (water1.getHybridZone() == CORE)
            continue;

        if (water1.getMoltype() != waterTypeValue)
            continue;

        for (auto &water2 : simBox.getMMMolecules())
        {
            if (water2.getMoltype() != waterTypeValue)
                continue;

            auto &oxygen1   = water1.getAtom(0);
            auto &oxygen2   = water2.getAtom(0);
            auto &hydrogen1 = water1.getAtom(1);
            auto &hydrogen2 = water1.getAtom(2);
            auto &hydrogen3 = water2.getAtom(1);
            auto &hydrogen4 = water2.getAtom(2);

            const auto singleInteraction =
                [&](Atom &atomA, Atom &atomB, const auto &nonCoulPairPtr)
            {
                if (nonCoulPairPtr)
                {
                    calculateSingleInteraction<QMChargeTag, MMChargeTag>(
                        atomA,
                        atomB,
                        coulombPotential,
                        rCutSquared,
                        simBox,
                        *nonCoulPairPtr,
                        totalCoulombEnergy,
                        totalNonCoulombEnergy
                    );
                }
            };

            // O-O interaction
            singleInteraction(oxygen1, oxygen2, state._nonCoulombPairOO);

            // O-H interactions
            singleInteraction(oxygen1, hydrogen3, state._nonCoulombPairOH);
            singleInteraction(oxygen1, hydrogen4, state._nonCoulombPairOH);
            singleInteraction(hydrogen1, oxygen2, state._nonCoulombPairOH);
            singleInteraction(hydrogen2, oxygen2, state._nonCoulombPairOH);

            // H-H interactions
            singleInteraction(hydrogen1, hydrogen3, state._nonCoulombPairHH);
            singleInteraction(hydrogen1, hydrogen4, state._nonCoulombPairHH);
            singleInteraction(hydrogen2, hydrogen3, state._nonCoulombPairHH);
            singleInteraction(hydrogen2, hydrogen4, state._nonCoulombPairHH);
        }
    }

    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
}

/**
 * @brief Compute outer-to-outer interactions by brute force.
 *
 * @param state Inter-water parameters.
 * @param simBox Simulation box containing molecules.
 * @param physicalData Physical data to store energy results.
 * @param coulombPotential Coulomb potential evaluator.
 * @param cellList Cell list structure (unused).
 */
void InterWaterStrategyBruteForce::calculateOuterToOuterForces(
    const InterWaterState                        &state,
    molsys::SimulationBox                        &simBox,
    PhysicalData                                 &physicalData,
    const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
    CellList                                     &cellList
)
{
    calculate(state, simBox, physicalData, coulombPotential, cellList);
}

/**
 * @brief Compute smoothing-zone interactions against MM molecules.
 *
 * @param state Inter-water parameters.
 * @param simBox Simulation box containing molecules.
 * @param physicalData Physical data to store energy results.
 * @param coulombPotential Coulomb potential evaluator.
 */
void InterWaterStrategyBruteForce::calculateHotspotSmoothingMMForces(
    const InterWaterState                        &state,
    molsys::SimulationBox                        &simBox,
    PhysicalData                                 &physicalData,
    const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
    CellList & /*cellList*/
)
{
    const auto rCut        = pot::CoulombPotential::getCoulombRadiusCutOff();
    const auto rCutSquared = rCut * rCut;

    auto totalCoulombEnergy    = 0.0;
    auto totalNonCoulombEnergy = 0.0;

    const auto waterTypeValue = simBox.getWaterType().value_or(size_t{0});

    for (auto &water1 : simBox.getMoleculesInsideZone(SMOOTHING))
    {
        if (water1.getMoltype() != waterTypeValue)
            continue;

        for (auto &water2 : simBox.getMoleculesOutsideZone(SMOOTHING))
        {
            if (water2.getMoltype() != waterTypeValue)
                continue;

            auto &oxygen1   = water1.getAtom(0);
            auto &oxygen2   = water2.getAtom(0);
            auto &hydrogen1 = water1.getAtom(1);
            auto &hydrogen2 = water1.getAtom(2);
            auto &hydrogen3 = water2.getAtom(1);
            auto &hydrogen4 = water2.getAtom(2);

            const auto singleInteraction =
                [&](Atom &atomA, Atom &atomB, const auto &nonCoulPairPtr)
            {
                if (nonCoulPairPtr)
                {
                    calculateSingleInteraction<MMChargeTag, QMChargeTag>(
                        atomA,
                        atomB,
                        coulombPotential,
                        rCutSquared,
                        simBox,
                        *nonCoulPairPtr,
                        totalCoulombEnergy,
                        totalNonCoulombEnergy
                    );
                }
            };

            // O-O interaction
            singleInteraction(oxygen1, oxygen2, state._nonCoulombPairOO);

            // O-H interactions
            singleInteraction(oxygen1, hydrogen3, state._nonCoulombPairOH);
            singleInteraction(oxygen1, hydrogen4, state._nonCoulombPairOH);
            singleInteraction(hydrogen1, oxygen2, state._nonCoulombPairOH);
            singleInteraction(hydrogen2, oxygen2, state._nonCoulombPairOH);

            // H-H interactions
            singleInteraction(hydrogen1, hydrogen3, state._nonCoulombPairHH);
            singleInteraction(hydrogen1, hydrogen4, state._nonCoulombPairHH);
            singleInteraction(hydrogen2, hydrogen3, state._nonCoulombPairHH);
            singleInteraction(hydrogen2, hydrogen4, state._nonCoulombPairHH);
        }
    }

    size_t i = 0;
    for (auto &water1 : simBox.getMoleculesInsideZone(SMOOTHING))
    {
        if (water1.getMoltype() != waterTypeValue)
        {
            ++i;
            continue;
        }

        size_t j = 0;
        for (auto &water2 : simBox.getMoleculesInsideZone(SMOOTHING))
        {
            if (water2.getMoltype() != waterTypeValue)
            {
                ++j;
                continue;
            }

            if (i == j)
            {
                ++j;
                continue;
            }

            auto &oxygen1   = water1.getAtom(0);
            auto &oxygen2   = water2.getAtom(0);
            auto &hydrogen1 = water1.getAtom(1);
            auto &hydrogen2 = water1.getAtom(2);
            auto &hydrogen3 = water2.getAtom(1);
            auto &hydrogen4 = water2.getAtom(2);

            const auto singleInteractionOneWay =
                [&](Atom &atomA, Atom &atomB, const auto &nonCoulPairPtr)
            {
                if (nonCoulPairPtr)
                {
                    calculateSingleInteractionOneWay<MMChargeTag, QMChargeTag>(
                        atomA,
                        atomB,
                        coulombPotential,
                        rCutSquared,
                        simBox,
                        *nonCoulPairPtr,
                        totalCoulombEnergy,
                        totalNonCoulombEnergy
                    );
                }
            };

            // clang-format off
            // O-O interaction
            singleInteractionOneWay(oxygen1, oxygen2, state._nonCoulombPairOO);

            // O-H interactions
            singleInteractionOneWay(oxygen1, hydrogen3, state._nonCoulombPairOH);
            singleInteractionOneWay(oxygen1, hydrogen4, state._nonCoulombPairOH);
            singleInteractionOneWay(hydrogen1, oxygen2, state._nonCoulombPairOH);
            singleInteractionOneWay(hydrogen2, oxygen2, state._nonCoulombPairOH);

            // H-H interactions
            singleInteractionOneWay(hydrogen1, hydrogen3, state._nonCoulombPairHH);
            singleInteractionOneWay(hydrogen1, hydrogen4, state._nonCoulombPairHH);
            singleInteractionOneWay(hydrogen2, hydrogen3, state._nonCoulombPairHH);
            singleInteractionOneWay(hydrogen2, hydrogen4, state._nonCoulombPairHH);
            //clang-format on

            ++j;
        }
        ++i;
    }

    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
}
