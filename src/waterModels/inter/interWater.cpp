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

#include <utility>
#include <vector>

#include "atom.hpp"                // for Atom
#include "coulombPotential.hpp"    // for CoulombPotential
#include "guffPair.hpp"            // for GuffPair
#include "physicalData.hpp"        // for PhysicalData
#include "potentialSettings.hpp"   // for PotentialSettings
#include "simulationBox.hpp"       // for SimulationBox
#include "typeAliases.hpp"
#include "vector3d.hpp"   // for normSquared

using namespace potential;
using namespace pq;
using namespace settings;
using namespace waterModel;

/**
 * @brief Construct an inert inter-water handler.
 *
 * @details Creates a default state and installs the null strategy so an
 * InterWater object can exist before a real water model is configured.
 */
InterWater::InterWater()
    : _state{}, _strategy{std::make_unique<InterWaterStrategyNull>()}
{
}

/**
 * @brief Construct an inter-water handler from a state and a strategy.
 *
 * @details Takes ownership of the supplied strategy, stores the provided
 * state, and initializes the GUFF pairs for the configured water model.
 *
 * @param state The inter-water parameters.
 * @param strategy The strategy object used to evaluate the interaction.
 */
InterWater::InterWater(
    InterWaterState                     state,
    std::unique_ptr<InterWaterStrategy> strategy
)
    : _state{std::move(state)}, _strategy{std::move(strategy)}
{
    initState();
}

/**
 * @brief Build the GUFF pairs for the configured inter-water model.
 *
 * @details Resolves the non-Coulomb cutoff, instantiates the three GUFF pair
 * objects, and finalizes their cutoff-dependent coefficients.
 */
void InterWater::initGuffPairs()
{
    _state._nonCoulombCutOff =
        PotentialSettings::getNonCoulombRadiusCutOff().value_or(
            PotentialSettings::getCoulombRadiusCutOff()
        );

    const auto makeGuffPair = [this](const std::vector<double> &coefficients)
    { return GuffPair{_state._nonCoulombCutOff, coefficients}; };

    const auto finalizeCutOff = [this](auto &guffPair)
    {
        const auto [energyCutOff, forceCutOff] =
            guffPair.calculate(_state._nonCoulombCutOff);
        guffPair.setEnergyCutOff(energyCutOff);
        guffPair.setForceCutOff(forceCutOff);
    };

    _state._guffPairOO = makeGuffPair(_state._guffCoefficientsOO);
    _state._guffPairOH = makeGuffPair(_state._guffCoefficientsOH);
    _state._guffPairHH = makeGuffPair(_state._guffCoefficientsHH);

    finalizeCutOff(_state._guffPairOO);
    finalizeCutOff(_state._guffPairOH);
    finalizeCutOff(_state._guffPairHH);
}

/**
 * @brief Precompute pairwise charge products for inter-water interactions.
 *
 * @details Caches O-O, O-H, and H-H charge products in the inter-water state
 * to avoid repeated multiplications during the interaction loops.
 */
void InterWater::initChargeProducts()
{
    const auto oxygenCharge   = _state._oxygenCharge;
    const auto hydrogenCharge = _state._hydrogenCharge;
    _state._chargeProductOO   = oxygenCharge * oxygenCharge;
    _state._chargeProductOH   = oxygenCharge * hydrogenCharge;
    _state._chargeProductHH   = hydrogenCharge * hydrogenCharge;
}

/**
 * @brief Evaluate intermolecular water interactions by brute force.
 *
 * @details Iterates over all active water-molecule pairs, accumulates Coulomb
 * and non-Coulomb contributions, and adds forces directly to the atoms.
 */
void InterWaterStrategyBruteForce::calculate(
    const InterWaterState  &state,
    SimBox                 &simBox,
    PhysicalData           &physicalData,
    const SharedCoulombPot &coulombPotential
)
{
    const auto chargeProductOO = state._chargeProductOO;
    const auto chargeProductOH = state._chargeProductOH;
    const auto chargeProductHH = state._chargeProductHH;

    const auto rCut        = CoulombPot::getCoulombRadiusCutOff();
    const auto rCutSquared = rCut * rCut;

    auto totalCoulombEnergy    = 0.0;
    auto totalNonCoulombEnergy = 0.0;

    size_t i = 0;
    for (auto &water1 : simBox.getWaterTypeMolecules())
    {
        if (!water1.isActive())
            continue;

        size_t j = 0;
        for (auto &water2 : simBox.getWaterTypeMolecules())
        {
            // avoid double counting and self interaction
            if (j >= i)
                break;

            if (!water2.isActive())
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

            const auto singleInteraction = [&](Atom           &atomA,
                                               Atom           &atomB,
                                               const double    chargeProduct,
                                               const GuffPair &guffPair)
            {
                calculateSingleInteraction(
                    atomA,
                    atomB,
                    chargeProduct,
                    coulombPotential,
                    rCutSquared,
                    simBox,
                    guffPair,
                    totalCoulombEnergy,
                    totalNonCoulombEnergy
                );
            };

            // clang-format off
            // O-O interaction
            singleInteraction(oxygen1, oxygen2, chargeProductOO, state._guffPairOO);

            // O-H interactions
            singleInteraction(oxygen1, hydrogen3, chargeProductOH, state._guffPairOH);
            singleInteraction(oxygen1, hydrogen4, chargeProductOH, state._guffPairOH);
            singleInteraction(oxygen2, hydrogen1, chargeProductOH, state._guffPairOH);
            singleInteraction(oxygen2, hydrogen2, chargeProductOH, state._guffPairOH);

            // H-H interactions
            singleInteraction(hydrogen1, hydrogen3, chargeProductHH, state._guffPairHH);
            singleInteraction(hydrogen1, hydrogen4, chargeProductHH, state._guffPairHH);
            singleInteraction(hydrogen2, hydrogen3, chargeProductHH, state._guffPairHH);
            singleInteraction(hydrogen2, hydrogen4, chargeProductHH, state._guffPairHH);
            // clang-format on

            ++j;
        }
        ++i;
    }

    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
}

/**
 * @brief Calculate Coulomb and non-Coulomb contributions for one atom pair.
 *
 * @details Applies periodic boundary conditions, computes the distance,
 * evaluates Coulomb potential if within the Coulomb cutoff, and evaluates
 * non-Coulomb if within the non-Coulomb cutoff. Returns the Coulomb and
 * non-Coulomb energy contributions; forces are accumulated directly on the
 * atoms.
 *
 * @param atom1 The first atom of the pair.
 * @param atom2 The second atom of the pair.
 * @param chargeProduct The product of the atomic charges (pre-computed
 * for efficiency).
 * @param coulombPotential The Coulomb potential evaluator.
 * @param rCutSquared The squared Coulomb cutoff distance.
 * @param simBox The simulation box for periodic boundary calculations.
 * @param guffPair The GUFF pair object for non-Coulomb evaluation.
 *
 * @return A pair<double, double> containing the Coulomb and non-Coulomb energy
 * contributions. Force is added directly to the atoms' force vectors.
 */
void InterWaterStrategy::calculateSingleInteraction(
    Atom                   &atom1,
    Atom                   &atom2,
    const double            chargeProduct,
    const SharedCoulombPot &coulombPotential,
    const double            rCutSquared,
    const SimBox           &simBox,
    const GuffPair         &guffPair,
    double                 &coulombEnergy,
    double                 &nonCoulombEnergy
)
{
    const auto xyz_i = atom1.getPosition();
    const auto xyz_j = atom2.getPosition();

    auto dxyz = xyz_i - xyz_j;

    const auto txyz = -simBox.calcShiftVector(dxyz);

    dxyz += txyz;

    const double distanceSquared = normSquared(dxyz);

    if (distanceSquared < rCutSquared)
    {
        const double distance = ::sqrt(distanceSquared);

        auto [e, f]    = coulombPotential->calculate(distance, chargeProduct);
        coulombEnergy += e;

        if (distance < guffPair.getRadialCutOff())
        {
            auto [nonCoulE, nonCoulF]  = guffPair.calculate(distance);
            nonCoulombEnergy          += nonCoulE;
            f                         += nonCoulF;
        }

        f                   /= distance;
        const auto forcexyz  = f * dxyz;

        const auto shiftForcexyz = forcexyz * txyz;

        atom1.addForce(forcexyz);
        atom2.addForce(-forcexyz);

        atom1.addShiftForce(shiftForcexyz);
    }
}