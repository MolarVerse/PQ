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
 * @brief Dispatch inter-water calculations via the active strategy.
 *
 * @param simBox Simulation box containing molecules.
 * @param physicalData Physical data to store energy results.
 * @param sharedCoulombPot Shared Coulomb potential used by the strategy.
 * @param cellList Cell list structure used for neighbor searching.
 */
void InterWater::calculate(
    pq::SimBox                 &simBox,
    pq::PhysicalData           &physicalData,
    const pq::SharedCoulombPot &sharedCoulombPot,
    pq::CellList               &cellList
)
{
    if (!_strategy)
        return;

    _strategy
        ->calculate(_state, simBox, physicalData, sharedCoulombPot, cellList);
}

/**
 * @brief Dispatch inter-water QMMM force calculations via the active strategy.
 *
 * @param simBox Simulation box containing molecules.
 * @param physicalData Physical data to store energy results.
 * @param sharedCoulombPot Shared Coulomb potential used by the strategy.
 * @param cellList Cell list structure used for neighbor searching.
 */
void InterWater::calculateQMMMForces(
    pq::SimBox                 &simBox,
    pq::PhysicalData           &physicalData,
    const pq::SharedCoulombPot &sharedCoulombPot,
    pq::CellList               &cellList
)
{
    if (!_strategy)
        return;

    _strategy->calculateCoreToOuterForces(
        _state,
        simBox,
        physicalData,
        sharedCoulombPot,
        cellList
    );

    _strategy->calculateLayerToOuterForces(
        _state,
        simBox,
        physicalData,
        sharedCoulombPot,
        cellList
    );

    _strategy->calculateOuterToOuterForces(
        _state,
        simBox,
        physicalData,
        sharedCoulombPot,
        cellList
    );
}

void InterWater::calculateHotspotSmoothingMMForces(
    pq::SimBox                 &simBox,
    pq::PhysicalData           &physicalData,
    const pq::SharedCoulombPot &sharedCoulombPot,
    pq::CellList               &cellList
)
{
    _strategy->calculateHotspotSmoothingMMForces(
        _state,
        simBox,
        physicalData,
        sharedCoulombPot,
        cellList
    );
}

/**
 * @brief Construct an inter-water handler from a state and a strategy.
 *
 * @details Takes ownership of the supplied strategy, stores the provided
 * state, and initializes the non-Coulomb pairs for the configured water
 * model.
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
 * @brief Apply radial cutoffs to configured inter-water non-Coulomb pairs.
 *
 * @details Uses the explicit non-Coulomb cutoff when configured, otherwise
 * falls back to the Coulomb cutoff. O-O is always updated, while O-H and H-H
 * are only updated when oxygen-only non-Coulomb interactions are disabled.
 */
void InterWater::setNonCoulombCutOffRadii()
{
    const auto radialCutOff =
        PotentialSettings::getNonCoulombRadiusCutOff().value_or(
            PotentialSettings::getCoulombRadiusCutOff()
        );

    const auto setCutOff = [radialCutOff](auto &nonCoulombPair)
    {
        if (nonCoulombPair)
            nonCoulombPair->setRadialCutOff(radialCutOff);
    };

    setCutOff(_state._nonCoulombPairOO);

    if (!_state._oxygenOnlyNonCoulomb)
    {
        setCutOff(_state._nonCoulombPairOH);
        setCutOff(_state._nonCoulombPairHH);
    }
}

/**
 * @brief Initialize the non-Coulomb pairs for the configured inter-water model.
 *
 * @details Sets up energy and force cutoff values for the three inter-water
 * non-Coulomb pairs (OO, OH, HH) by evaluating them at their radial cutoff
 * distances.
 */
void InterWater::initNonCoulombPairs()
{
    const auto setForceAndEnergyCutOff = [](auto &nonCoulPair)
    {
        if (!nonCoulPair)
            return;
        const auto [energyCutOff, forceCutOff] =
            nonCoulPair->calculate(nonCoulPair->getRadialCutOff());
        nonCoulPair->setEnergyCutOff(energyCutOff);
        nonCoulPair->setForceCutOff(forceCutOff);
    };

    setForceAndEnergyCutOff(_state._nonCoulombPairOO);
    setForceAndEnergyCutOff(_state._nonCoulombPairOH);
    setForceAndEnergyCutOff(_state._nonCoulombPairHH);
}
