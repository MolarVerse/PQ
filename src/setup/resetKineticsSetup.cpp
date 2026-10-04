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

#include "resetKineticsSetup.hpp"

#include <format>

#include "engine.hpp"
#include "generalSettings.hpp"
#include "mdEngine.hpp"
#include "resetKinetics.hpp"
#include "resetKineticsSettings.hpp"
#include "timingsSettings.hpp"

namespace setup
{

    /**
     * @brief constructs a new Reset Kinetics Setup:: Reset Kinetics Setup
     * object and calls setup
     *
     * @param engine
     */
    void setupResetKinetics(engine::Engine &engine)
    {
        if (!settings::GeneralSettings::isMDJobType())
            return;

        out::StdoutOutput::writeSetup("Reset Kinetics");
        engine.getLogOutput().writeSetup("Reset Kinetics");

        ResetKineticsSetup resetKineticsSetup(
            dynamic_cast<engine::MDEngine &>(engine)
        );
        resetKineticsSetup.setup();
    }

    /**
     * @brief Construct a new Reset Kinetics Setup object
     *
     * @param engine
     */
    ResetKineticsSetup::ResetKineticsSetup(engine::MDEngine &engine)
        : _engine(engine)
    {
    }

    /**
     * @brief setup nscale, fscale, nreset, freset
     *
     * @details decides if temperature and momentum or only temperature is reset
     * It checks if either fscale or freset is set to 0 and sets it to the
     * number of steps + 1, so that the reset is not performed. nreset and
     * freset are set to 0 if they are not set.
     *
     */
    void ResetKineticsSetup::setup()
    {
        _engine.makeResetKinetics(::resetKinetics::ResetKinetics(
            nScale,
            fScale,
            nReset,
            fReset,
            nResetAngular,
            fResetAngular,
            fResetForces
        );

        writeSetupInfo();
    }

    /**
     * @brief writes setup info to log file
     */
    void ResetKineticsSetup::writeSetupInfo() const
    {
        const auto _fScale = settings::ResetKineticsSettings::getFScale();
        const auto _fReset = settings::ResetKineticsSettings::getFReset();
        const auto _fResetAngular =
            settings::ResetKineticsSettings::getFResetAngular();
        const auto _fResetForces =
            settings::ResetKineticsSettings::getFResetForces();

        const int fScale = _fScale == 0 ? -1 : static_cast<int>(_fScale);
        const int fReset = _fReset == 0 ? -1 : static_cast<int>(_fReset);
        const int fResetAngular =
            _fResetAngular == 0 ? -1 : static_cast<int>(_fResetAngular);
        const int fResetForces =
            _fResetForces == 0 ? -1 : static_cast<int>(_fResetForces);

        const auto nScale = settings::ResetKineticsSettings::getNScale();
        const auto nReset = settings::ResetKineticsSettings::getNReset();
        const auto nResetAngular =
            settings::ResetKineticsSettings::getNResetAngular();

        const auto nScaleMsg = std::format("first {:5d} steps,", nScale);
        const auto fScaleMsg = std::format("every {:5d} steps", fScale);
        const auto nResetMsg = std::format("first {:5d} steps,", nReset);
        const auto fResetMsg = std::format("every {:5d} steps", fReset);
        const auto nResetAngMsg =
            std::format("first {:5d} steps,", nResetAngular);
        const auto fResetAngMsg =
            std::format("every {:5d} steps", fResetAngular);
        const auto fResetForcesMsg =
            std::format("every {:5d} steps", fResetForces);

        const auto scaleMsg =
            std::format("reset temperature:      {} {}", nScaleMsg, fScaleMsg);
        const auto resetMsg =
            std::format("reset momentum:         {} {}", nResetMsg, fResetMsg);
        const auto resetAngMsg = std::format(
            "reset angular momentum: {} {}",
            nResetAngMsg,
            fResetAngMsg
        );
        const auto resetForceMsg =
            std::format("reset forces:           {}   ", fResetForcesMsg);

        auto &log = _engine.getLogOutput();

        log.writeSetupInfo(scaleMsg);
        log.writeSetupInfo(resetMsg);
        log.writeSetupInfo(resetAngMsg);
        log.writeSetupInfo(resetForceMsg);
        log.writeEmptyLine();
    }

}   // namespace setup
