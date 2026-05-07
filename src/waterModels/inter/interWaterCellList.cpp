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

#include <utility>
#include <vector>

#include "atom.hpp"               // for Atom
#include "coulombPotential.hpp"   // for CoulombPotential
#include "interWater.hpp"         // for InterWater
#include "physicalData.hpp"       // for PhysicalData
#include "simulationBox.hpp"      // for SimulationBox
#include "typeAliases.hpp"

using namespace pq;
using namespace waterModel;

/**
 * @brief Evaluate intermolecular water interactions via cell list.
 *
 */
void InterWaterStrategyCellList::calculate(
    const InterWaterState  &state,
    SimBox                 &simBox,
    PhysicalData           &physicalData,
    const SharedCoulombPot &coulombPotential,
    CellList               &cellList
)
{
    const auto chargeProductOO = state._chargeProductOO;
    const auto chargeProductOH = state._chargeProductOH;
    const auto chargeProductHH = state._chargeProductHH;

    const auto rCut        = CoulombPot::getCoulombRadiusCutOff();
    const auto rCutSquared = rCut * rCut;

    auto totalCoulombEnergy    = 0.0;
    auto totalNonCoulombEnergy = 0.0;

    const auto waterType = simBox.getWaterType();

    for (const auto &cell_i : cellList.getCells())
    {
        size_t i = 0;
        for (auto &water1 : cell_i.getWaterTypeMolecules(waterType))
        {
            if (!water1->isActive())
                continue;

            size_t j = 0;
            for (auto &water2 : cell_i.getWaterTypeMolecules(waterType))
            {
                // avoid double counting and self interaction
                if (j >= i)
                    break;

                if (!water2->isActive())
                {
                    ++j;
                    continue;
                }

                auto &oxygen1   = water1->getAtom(0);
                auto &oxygen2   = water2->getAtom(0);
                auto &hydrogen1 = water1->getAtom(1);
                auto &hydrogen2 = water1->getAtom(2);
                auto &hydrogen3 = water2->getAtom(1);
                auto &hydrogen4 = water2->getAtom(2);

                const auto singleInteraction = [&](Atom        &atomA,
                                                   Atom        &atomB,
                                                   const double chargeProduct,
                                                   const auto  &nonCoulPairPtr)
                {
                    if (nonCoulPairPtr)
                        calculateSingleInteraction(
                            atomA,
                            atomB,
                            chargeProduct,
                            coulombPotential,
                            rCutSquared,
                            simBox,
                            *nonCoulPairPtr,
                            totalCoulombEnergy,
                            totalNonCoulombEnergy
                        );
                };

                // clang-format off
                // O-O interaction
                singleInteraction(oxygen1, oxygen2, chargeProductOO, state._nonCoulombPairOO);

                // O-H interactions
                singleInteraction(oxygen1, hydrogen3, chargeProductOH, state._nonCoulombPairOH);
                singleInteraction(oxygen1, hydrogen4, chargeProductOH, state._nonCoulombPairOH);
                singleInteraction(oxygen2, hydrogen1, chargeProductOH, state._nonCoulombPairOH);
                singleInteraction(oxygen2, hydrogen2, chargeProductOH, state._nonCoulombPairOH);

                // H-H interactions
                singleInteraction(hydrogen1, hydrogen3, chargeProductHH, state._nonCoulombPairHH);
                singleInteraction(hydrogen1, hydrogen4, chargeProductHH, state._nonCoulombPairHH);
                singleInteraction(hydrogen2, hydrogen3, chargeProductHH, state._nonCoulombPairHH);
                singleInteraction(hydrogen2, hydrogen4, chargeProductHH, state._nonCoulombPairHH);
                // clang-format on

                ++j;
            }
            ++i;
        }
    }

    for (const auto &cell1 : cellList.getCells())
        for (const auto *cell2 : cell1.getNeighbourCells())
            for (auto &water1 : cell1.getWaterTypeMolecules(waterType))
            {
                if (!water1->isActive())
                    continue;

                for (auto &water2 : cell2->getWaterTypeMolecules(waterType))
                {
                    // avoid self interaction
                    if (water1 == water2)
                        continue;

                    if (!water2->isActive())
                        continue;

                    auto &oxygen1   = water1->getAtom(0);
                    auto &oxygen2   = water2->getAtom(0);
                    auto &hydrogen1 = water1->getAtom(1);
                    auto &hydrogen2 = water1->getAtom(2);
                    auto &hydrogen3 = water2->getAtom(1);
                    auto &hydrogen4 = water2->getAtom(2);

                    const auto singleInteraction =
                        [&](Atom        &atomA,
                            Atom        &atomB,
                            const double chargeProduct,
                            const auto  &nonCoulPairPtr)
                    {
                        if (nonCoulPairPtr)
                            calculateSingleInteraction(
                                atomA,
                                atomB,
                                chargeProduct,
                                coulombPotential,
                                rCutSquared,
                                simBox,
                                *nonCoulPairPtr,
                                totalCoulombEnergy,
                                totalNonCoulombEnergy
                            );
                    };

                    // clang-format off
                    // O-O interaction
                    singleInteraction(oxygen1, oxygen2, chargeProductOO, state._nonCoulombPairOO);

                    // O-H interactions
                    singleInteraction(oxygen1, hydrogen3, chargeProductOH, state._nonCoulombPairOH);
                    singleInteraction(oxygen1, hydrogen4, chargeProductOH, state._nonCoulombPairOH);
                    singleInteraction(oxygen2, hydrogen1, chargeProductOH, state._nonCoulombPairOH);
                    singleInteraction(oxygen2, hydrogen2, chargeProductOH, state._nonCoulombPairOH);

                    // H-H interactions
                    singleInteraction(hydrogen1, hydrogen3, chargeProductHH, state._nonCoulombPairHH);
                    singleInteraction(hydrogen1, hydrogen4, chargeProductHH, state._nonCoulombPairHH);
                    singleInteraction(hydrogen2, hydrogen3, chargeProductHH, state._nonCoulombPairHH);
                    singleInteraction(hydrogen2, hydrogen4, chargeProductHH, state._nonCoulombPairHH);
                    // clang-format on
                }
            }

    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
}
