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

#include "potentialSetup.hpp"

#include "angleForceField.hpp"           // for potential
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "coulombWolf.hpp"               // for CoulombWolf
#include "engine.hpp"                    // for Engine
#include "exceptions.hpp"                // for ParameterFileException
#include "forceFieldClass.hpp"           // for ForceField
#include "forceFieldNonCoulomb.hpp"      // for ForceFieldNonCoulomb
#include "guffNonCoulomb.hpp"            // for GuffNonCoulomb
#include "nonCoulombPair.hpp"            // IWYU pragma: keep for NonCoulombPair
#include "nonCoulombPotential.hpp"       // for NonCoulombPotential
#include "potential.hpp"                 // for Potential
#include "potentialSettings.hpp"         // for PotentialSettings
#include "simulationBox.hpp"             // for SimulationBox

#include <algorithm>     // for __for_each_fn, __sort_fn
#include <functional>    // for identity
#include <memory>        // for swap, shared_ptr, __shared_ptr_access
#include <string>        // for operator==
#include <string_view>   // for string_view
#include <vector>        // for vector

using namespace setup;
using namespace potential;

/**
 * @brief wrapper to create PotentialSetup object and call setup
 *
 * @param engine
 */
void setup::setupPotential(engine::Engine &engine)
{
    engine.getStdoutOutput().writeSetup("MM potential");
    engine.getLogOutput().writeSetup("MM potential");

    PotentialSetup potentialSetup(engine);
    potentialSetup.setup();
}

/**
 * @brief sets all nonBonded potential types
 *
 * @details if forceFieldNonCoulombics are activated it sets up also the nonCoulombic pairs
 *
 */
void PotentialSetup::setup()
{
    setupCoulomb();
    setupNonCoulomb();

    if (!_engine.isForceFieldNonCoulombicsActivated())
        return;

    setupNonCoulombicPairs();
}

/**
 * @brief sets coulomb potential type
 *
 * @details possible types are:
 * 1) none (shifted coulomb potential)
 * 2) wolf (wolf long range correction)
 *
 * @param coulombType
 */
void PotentialSetup::setupCoulomb()
{
    const auto coulombRadiusCutOff = settings::PotentialSettings::getCoulombRadiusCutOff();
    auto       wolfParameter       = settings::PotentialSettings::getWolfParameter();

    if (settings::PotentialSettings::getCoulombLongRangeType() == "wolf")
        _engine.getPotential().makeCoulombPotential(CoulombWolf(coulombRadiusCutOff, wolfParameter));
    else
        _engine.getPotential().makeCoulombPotential(CoulombShiftedPotential(coulombRadiusCutOff));
}

/**
 * @brief sets nonCoulomb potential type
 *
 * @details decides wether to use Guff or ForceFieldNonCoulomb potential
 *
 */
void PotentialSetup::setupNonCoulomb()
{
    if (_engine.getForceFieldPtr()->isNonCoulombicActivated())
    {
        // _engine.getPotential().makeNonCoulombPotential(potential::ForceFieldNonCoulomb()); TODO: think of a clever way to do
        // this
    }
    else
        _engine.getPotential().makeNonCoulombPotential(potential::GuffNonCoulomb());
}

/**
 * @brief sets up nonCoulombic pairs in the ForceFieldNonCoulomb potential
 *
 * @details Following steps are performed:
 * 1) calculate energy and force cut off for each nonCoulombic pair
 * 2) determine internal global vdw types
 * 3) check if all self interacting non coulombics are set
 * 4) sort self interacting non coulombics
 * 5) check if all self interacting non coulombics are set
 * 6) fill diagonal elements of nonCoulombicPairsMatrix
 * 7) fill non diagonal elements of nonCoulombicPairsMatrix
 *
 * @throws customException::ParameterFileException if not all self interacting non coulombics are set
 *
 */
void PotentialSetup::setupNonCoulombicPairs()
{
    auto &potential = dynamic_cast<ForceFieldNonCoulomb &>(_engine.getPotential().getNonCoulombPotential());

    potential.setupNonCoulombicCutoffs();

    _engine.getSimulationBox().setupExternalToInternalGlobalVdwTypesMap();
    potential.determineInternalGlobalVdwTypes(_engine.getSimulationBox().getExternalToInternalGlobalVDWTypes());

    const auto numberOfGlobalVdwTypes           = _engine.getSimulationBox().getExternalGlobalVdwTypes().size();
    auto       selfInteractionNonCoulombicPairs = potential.getSelfInteractionNonCoulombicPairs();

    if (selfInteractionNonCoulombicPairs.size() != numberOfGlobalVdwTypes)
        throw customException::ParameterFileException(
            "Not all self interacting non coulombics were set in the noncoulombics section of the parameter file");

    std::ranges::sort(selfInteractionNonCoulombicPairs,
                      [](const auto &nonCoulombicPair1, const auto &nonCoulombicPair2)
                      { return nonCoulombicPair1->getInternalType1() < nonCoulombicPair2->getInternalType1(); });

    potential.fillDiagonalElementsOfNonCoulombPairsMatrix(selfInteractionNonCoulombicPairs);
    potential.fillOffDiagonalElementsOfNonCoulombPairsMatrix();
}