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

#include <algorithm>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "coulombReactionField.hpp"
#include "coulombShiftedPotential.hpp"
#include "coulombWolf.hpp"
#include "engine.hpp"
#include "exceptions.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "forceFieldSettings.hpp"
#include "guffNonCoulomb.hpp"
#include "potential.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"

namespace setup
{

    /**
     * @brief wrapper to create PotentialSetup object and call setup
     *
     * @param engine
     */
    void setupPotential(engine::Engine &engine)
    {
        out::StdoutOutput::writeSetup("MM potential");
        engine.getLogOutput().writeSetup("MM potential");

        PotentialSetup potentialSetup(engine);
        potentialSetup.setup();
    }

    /**
     * @brief Construct a new Potential Setup:: Potential Setup object
     *
     * @param engine
     */
    PotentialSetup::PotentialSetup(engine::Engine &engine) : _engine(engine) {}

    /**
     * @brief sets all nonBonded potential types
     *
     * @details if forceFieldNonCoulombics are activated it sets up also the
     * nonCoulombic pairs
     *
     * @note the non-Coulomb potential type itself is set up earlier, via
     * setupNonCoulombPotentialType(), before the parameter file is read -
     * see setupRequestedJob(). Re-creating it here would discard the
     * nonCoulombic pairs already read from the parameter file.
     *
     */
    void PotentialSetup::setup()
    {
        setupCoulomb();

        if (settings::ForceFieldSettings::isNonCoulombicActive())
            setupNonCoulombicPairs();

        writeSetupInfo();
    }

    /**
     * @brief wrapper to create the non-Coulomb potential of the correct
     * concrete type before any files are read
     *
     * @details the parameter file reader needs to dynamic_cast the
     * non-Coulomb potential to ForceFieldNonCoulomb while reading the
     * NONCOULOMBICS section, so the potential has to already have its
     * final concrete type by the time readFiles() runs.
     *
     * @param engine
     */
    void setupNonCoulombPotentialType(engine::Engine &engine)
    {
        PotentialSetup potentialSetup(engine);
        potentialSetup.setupNonCoulomb();
    }

    /**
     * @brief sets coulomb potential type
     *
     * @details possible types are:
     * 1) none (shifted coulomb potential)
     * 2) reaction field long range correction
     * 2) wolf long range correction
     *
     */
    void PotentialSetup::setupCoulomb()
    {
        const auto coulRCut =
            settings::PotentialSettings::getCoulombRadiusCutOff();
        const auto wolfParam = settings::PotentialSettings::getWolfParameter();
        const auto rfEpsilon =
            settings::PotentialSettings::getReactionFieldEpsilon();
        const auto &potential = _engine.getPotential();

        switch (settings::PotentialSettings::getCoulombLongRangeType())
        {
            using enum CoulombLongRangeType;

            case REACTION_FIELD:
                potential->makeCoulombPotential(
                    pot::CoulombReactionField(coulRCut, rfEpsilon)
                );
                return;

            case WOLF:
                potential->makeCoulombPotential(
                    pot::CoulombWolf(coulRCut, wolfParam)
                );
                return;

            case SHIFTED: break;
        }

        potential->makeCoulombPotential(pot::CoulombShiftedPotential(coulRCut));
    }

    /**
     * @brief sets nonCoulomb potential type
     *
     * @details decides wether to use Guff or ForceFieldNonCoulomb potential
     *
     */
    void PotentialSetup::setupNonCoulomb()
    {
        const auto &potential = _engine.getPotential();

        // NOTE: no else branch needed ForceFieldNonCoulomb is default
        //       makeForceFieldNonCoulomb is a no-op if already set
        //       However, it does also throw errors atm - thus the else
        //       statement is left out
        if (!settings::ForceFieldSettings::isNonCoulombicActive())
            potential->makeNonCoulombPotential(pot::GuffNonCoulomb());
        else
            potential->makeNonCoulombPotential(pot::ForceFieldNonCoulomb());
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
        const auto &pot    = _engine.getPotential();
        auto       &simBox = _engine.getSimulationBox();

        auto &nonCoulPot = dynamic_cast<pot::ForceFieldNonCoulomb &>(
            pot->getNonCoulombPotential()
        );
        nonCoulPot.setupNonCoulombicCutoffs();

        const auto &extToIntVDWTypes =
            simBox.getExternalToInternalGlobalVDWTypes();

        simBox.setupExternalToInternalGlobalVdwTypesMap();
        nonCoulPot.determineInternalGlobalVdwTypes(extToIntVDWTypes);

        const auto nGlobalVdwTypes = simBox.getExternalGlobalVdwTypes().size();
        auto selfNonCoulPairs = nonCoulPot.getSelfInteractionNonCoulPairs();

        if (selfNonCoulPairs.size() != nGlobalVdwTypes)
        {
            throw exc::ParameterFileException(
                "Not all self interacting non coulombics were set in the "
                "noncoulombics section of the parameter file"
            );
        }

        std::ranges::sort(
            selfNonCoulPairs,
            [](const auto &nonCoulombicPair1, const auto &nonCoulombicPair2)
            {
                const auto &internalType1 =
                    nonCoulombicPair1->getInternalType1();
                const auto &internalType2 =
                    nonCoulombicPair2->getInternalType1();
                return internalType1 < internalType2;
            }
        );

        nonCoulPot.fillDiagOfNonCoulPairsMatrix(selfNonCoulPairs);
        nonCoulPot.fillOffDiagOfNonCoulPairsMatrix();
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

        const auto coulLRType =
            settings::PotentialSettings::getCoulombLongRangeType();

        log.writeSetupInfo(
            std::format(
                "Coulomb long range type: {}",
                CoulombLongRangeTypeMeta::toString(coulLRType)
            )
        );
        log.writeEmptyLine();

        const auto coulRCut =
            settings::PotentialSettings::getCoulombRadiusCutOff();
        auto wolfParam = 0.0;
        auto rfEpsilon = 0.0;

        if (coulLRType == CoulombLongRangeType::WOLF)
            wolfParam = settings::PotentialSettings::getWolfParameter();

        if (coulLRType == CoulombLongRangeType::REACTION_FIELD)
            rfEpsilon = settings::PotentialSettings::getReactionFieldEpsilon();

        const auto coulRCutStr =
            std::format("Coulomb radius cut-off: {}", coulRCut);
        log.writeSetupInfo(coulRCutStr);

        if (coulLRType == CoulombLongRangeType::WOLF)
        {
            const auto wolfParamStr =
                std::format("Wolf parameter:         {}", wolfParam);
            log.writeSetupInfo(wolfParamStr);
        }
        else if (coulLRType == CoulombLongRangeType::REACTION_FIELD)
        {
            const auto rfEpsilonStr = std::format(
                "Reaction-field static relative permittivity: {}",
                rfEpsilon
            );
            log.writeSetupInfo(rfEpsilonStr);
        }

        log.writeEmptyLine();
    }

    /**
     * @brief writes non-coulomb potential setup information to log file
     *
     */
    void PotentialSetup::writeNonCoulombInfo() const
    {
        auto &log = _engine.getLogOutput();

        if (settings::ForceFieldSettings::isNonCoulombicActive())
        {
            auto      &simBox = _engine.getSimulationBox();
            const auto nGlobalVdwTypes =
                simBox.getExternalGlobalVdwTypes().size();

            log.writeSetupInfo(
                std::format("Non-coulombic potential: ForceField")
            );
            log.writeSetupInfo(
                std::format("Total Global VDW types:  {}", nGlobalVdwTypes)
            );
        }
        else
        {
            log.writeSetupInfo("Non-coulombic potential: Guff");
        }

        log.writeEmptyLine();
    }

}   // namespace setup
