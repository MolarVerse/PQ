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

#include "engine.hpp"          // for Engine
#include "mdEngine.hpp"        // for MDEngine
#include "resetKinetics.hpp"   // for ResetMomentum, ResetTemperature, resetK...
#include "resetKineticsSettings.hpp"   // for ResetKineticsSettings
#include "settings.hpp"                // for Settings
#include "timingsSettings.hpp"         // for TimingsSettings

using setup::resetKinetics::ResetKineticsSetup;
using namespace engine;
using namespace settings;

/**
 * @brief constructs a new Reset Kinetics Setup:: Reset Kinetics Setup object
 * and calls setup
 *
 * @param engine
 */
void setup::resetKinetics::setupResetKinetics(Engine &engine)
{
    if (!Settings::isMDJobType())
        return;

    engine.getStdoutOutput().writeSetup("Reset Kinetics");
    engine.getLogOutput().writeSetup("Reset Kinetics");

    ResetKineticsSetup resetKineticsSetup(dynamic_cast<MDEngine &>(engine));
    resetKineticsSetup.setup();
}

/**
 * @brief Construct a new Reset Kinetics Setup object
 *
 * @param engine
 */
ResetKineticsSetup::ResetKineticsSetup(MDEngine &engine) : _engine(engine){};

/**
 * @brief setup nscale, fscale, nreset, freset
 *
 * @details decides if temperature and momentum or only temperature is reset
 * It checks if either fscale or freset is set to 0 and sets it to the number of
 * steps + 1, so that the reset is not performed. nreset and freset are set to 0
 * if they are not set.
 *
 */
void ResetKineticsSetup::setup()
{
    const auto nScale        = ResetKineticsSettings::getNScale();
    auto       fScale        = ResetKineticsSettings::getFScale();
    const auto nReset        = ResetKineticsSettings::getNReset();
    auto       fReset        = ResetKineticsSettings::getFReset();
    const auto nResetAngular = ResetKineticsSettings::getNResetAngular();
    auto       fResetAngular = ResetKineticsSettings::getFResetAngular();

    const auto numberOfSteps = TimingsSettings::getNumberOfSteps();

    fScale        = (0 == fScale) ? numberOfSteps + 1 : fScale;
    fReset        = (0 == fReset) ? numberOfSteps + 1 : fReset;
    fResetAngular = (0 == fResetAngular) ? numberOfSteps + 1 : fResetAngular;

    _engine.getResetKinetics() = ::resetKinetics::ResetKinetics(
        nScale,
        fScale,
        nReset,
        fReset,
        nResetAngular,
        fResetAngular
    );

    writeSetupInfo();
}

/**
 * @brief writes setup info to log file
 */
void ResetKineticsSetup::writeSetupInfo() const
{
    const auto _fScale        = ResetKineticsSettings::getFScale();
    const auto _fReset        = ResetKineticsSettings::getFReset();
    const auto _fResetAngular = ResetKineticsSettings::getFResetAngular();

    const int fScale        = _fScale == 0 ? -1 : _fScale;
    const int fReset        = _fReset == 0 ? -1 : _fReset;
    const int fResetAngular = _fResetAngular == 0 ? -1 : _fResetAngular;

    const auto nScale        = ResetKineticsSettings::getNScale();
    const auto nReset        = ResetKineticsSettings::getNReset();
    const auto nResetAngular = ResetKineticsSettings::getNResetAngular();

    const auto nScaleMsg    = std::format("first {:5d} steps,", nScale);
    const auto fScaleMsg    = std::format("every {:5d} steps", fScale);
    const auto nResetMsg    = std::format("first {:5d} steps,", nReset);
    const auto fResetMsg    = std::format("every {:5d} steps", fReset);
    const auto nResetAngMsg = std::format("first {:5d} steps,", nResetAngular);
    const auto fResetAngMsg = std::format("every {:5d} steps", fResetAngular);

    // clang-format off
    const auto scaleMsg    = std::format("reset temperature:      {} {}", nScaleMsg, fScaleMsg);
    const auto resetMsg    = std::format("reset momentum:         {} {}", nResetMsg, fResetMsg);
    const auto resetAngMsg = std::format("reset angular momentum: {} {}", nResetAngMsg, fResetAngMsg);
    // clang-format on

    auto &log = _engine.getLogOutput();

    log.writeSetupInfo(scaleMsg);
    log.writeSetupInfo(resetMsg);
    log.writeSetupInfo(resetAngMsg);
    log.writeEmptyLine();
}