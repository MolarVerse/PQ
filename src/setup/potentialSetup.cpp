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

#include <algorithm>     // for __for_each_fn, __sort_fn
#include <functional>    // for identity
#include <memory>        // for swap, shared_ptr, __shared_ptr_access
#include <string>        // for operator==
#include <string_view>   // for string_view
#include <vector>        // for vector

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

using namespace setup;
using namespace potential;
using namespace engine;
using namespace settings;
using namespace customException;

/**
 * @brief wrapper to create PotentialSetup object and call setup
 *
 * @param engine
 */
void setup::setupPotential(Engine &engine)
{
    engine.getStdoutOutput().writeSetup("MM potential");
    engine.getLogOutput().writeSetup("MM potential");

    PotentialSetup potentialSetup(engine);
    potentialSetup.setup();
}

/**
 * @brief Construct a new Potential Setup:: Potential Setup object
 *
 * @param engine
 */
PotentialSetup::PotentialSetup(Engine &engine) : _engine(engine){};

/**
 * @brief sets all nonBonded potential types
 *
 * @details if forceFieldNonCoulombics are activated it sets up also the
 * nonCoulombic pairs
 *
 */
void PotentialSetup::setup()
{
    setupCoulomb();
    setupNonCoulomb();

    if (!_engine.isForceFieldNonCoulombicsActivated())
        return;

    setupNonCoulombicPairs();

    writeSetupInfo();
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
    const auto coulRCut  = PotentialSettings::getCoulombRadiusCutOff();
    const auto wolfParam = PotentialSettings::getWolfParameter();
    auto      &potential = _engine.getPotential();

    switch (PotentialSettings::getCoulombLongRangeType())
    {
        using enum CoulombLongRangeType;

        case WOLF:
            potential.makeCoulombPotential(CoulombWolf(coulRCut, wolfParam));
            break;

        case SHIFTED:
        default:
            potential.makeCoulombPotential(CoulombShiftedPotential(coulRCut));
    }
}

/**
 * @brief sets nonCoulomb potential type
 *
 * @details decides wether to use Guff or ForceFieldNonCoulomb potential
 *
 */
void PotentialSetup::setupNonCoulomb()
{
    auto &potential = _engine.getPotential();

    // NOTE: no else branch needed ForceFieldNonCoulomb is default
    //       makeForceFieldNonCoulomb is a no-op if already set
    //       However, it does also throw errors atm - thus the else
    //       statement is left out
    if (!_engine.getForceFieldPtr()->isNonCoulombicActivated())
        potential.makeNonCoulombPotential(GuffNonCoulomb());
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
 * @throws ParameterFileException if not all self interacting
 * non coulombics are set
 *
 */
void PotentialSetup::setupNonCoulombicPairs()
{
    auto &pot    = _engine.getPotential();
    auto &simBox = _engine.getSimulationBox();

    // clang-format off
    auto &nonCoulPot = dynamic_cast<pq::FFNonCoulomb &>(pot.getNonCoulombPotential());
    nonCoulPot.setupNonCoulombicCutoffs();
    // clang-format on

    const auto &extToIntVDWTypes = simBox.getExternalToInternalGlobalVDWTypes();

    simBox.setupExternalToInternalGlobalVdwTypesMap();
    nonCoulPot.determineInternalGlobalVdwTypes(extToIntVDWTypes);

    const auto nGlobalVdwTypes = simBox.getExternalGlobalVdwTypes().size();
    auto selfNonCoulPairs = nonCoulPot.getSelfInteractionNonCoulombicPairs();

    if (selfNonCoulPairs.size() != nGlobalVdwTypes)
        throw ParameterFileException(
            "Not all self interacting non coulombics were set in the "
            "noncoulombics section of the parameter file"
        );

    std::ranges::sort(
        selfNonCoulPairs,
        [](const auto &nonCoulombicPair1, const auto &nonCoulombicPair2)
        {
            const auto &internalType1 = nonCoulombicPair1->getInternalType1();
            const auto &internalType2 = nonCoulombicPair2->getInternalType1();
            return internalType1 < internalType2;
        }
    );

    nonCoulPot.fillDiagonalElementsOfNonCoulombPairsMatrix(selfNonCoulPairs);
    nonCoulPot.fillOffDiagonalElementsOfNonCoulombPairsMatrix();
}

/**
 * @brief writes setup information to log file
 *
 */
void PotentialSetup::writeSetupInfo() const
{
    writeCoulombInfo();
    writeNonCoulombInfo();
}

/**
 * @brief writes coulomb potential setup information to log file
 *
 */
void PotentialSetup::writeCoulombInfo() const
{
    auto &log = _engine.getLogOutput();

    const auto coulLRType = PotentialSettings::getCoulombLongRangeType();

    // clang-format off
    log.writeSetupInfo(std::format("Coulomb long range type: {}", string(coulLRType)));
    log.writeEmptyLine();
    // clang-format on

    const auto coulRCut  = PotentialSettings::getCoulombRadiusCutOff();
    auto       wolfParam = 0.0;

    if (coulLRType == CoulombLongRangeType::WOLF)
        wolfParam = PotentialSettings::getWolfParameter();

    // clang-format off
    const auto coulRCutStr  = std::format("Coulomb radius cut-off: {}", coulRCut);
    const auto wolfParamStr = std::format("Wolf parameter:         {}", wolfParam);
    // clang-format on

    log.writeSetupInfo(coulRCutStr);
    if (coulLRType == CoulombLongRangeType::WOLF)
        log.writeSetupInfo(wolfParamStr);
    log.writeEmptyLine();
}

/**
 * @brief writes non-coulomb potential setup information to log file
 *
 */
void PotentialSetup::writeNonCoulombInfo() const
{
    auto &log = _engine.getLogOutput();

    if (_engine.getForceFieldPtr()->isNonCoulombicActivated())
    {
        auto      &simBox          = _engine.getSimulationBox();
        const auto nGlobalVdwTypes = simBox.getExternalGlobalVdwTypes().size();

        // clang-format off
        log.writeSetupInfo(std::format("Non-coulombic potential: ForceField"));
        log.writeSetupInfo(std::format("Total Global VDW types:  {}", nGlobalVdwTypes));
        // clang-format on
    }
    else
        log.writeSetupInfo("Non-coulombic potential: Guff");
}
