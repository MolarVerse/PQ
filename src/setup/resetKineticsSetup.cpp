/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "engine.hpp"                  // for Engine
#include "resetKinetics.hpp"           // for ResetMomentum, ResetTemperature, resetK...
#include "resetKineticsSettings.hpp"   // for ResetKineticsSettings
#include "timingsSettings.hpp"         // for TimingsSettings

using namespace setup;

/**
 * @brief constructs a new Reset Kinetics Setup:: Reset Kinetics Setup object and calls setup
 *
 * @param engine
 */
void setup::setupResetKinetics(engine::Engine &engine)
{
    engine.getStdoutOutput().writeSetup("Reset Kinetics");
    engine.getLogOutput().writeSetup("Reset Kinetics");

    ResetKineticsSetup resetKineticsSetup(engine);
    resetKineticsSetup.setup();
}

/**
 * @brief setup nscale, fscale, nreset, freset
 *
 * @details decides if temperature and momentum or only temperature is reset
 * It checks if either fscale or freset is set to 0 and sets it to the number of steps + 1, so that the reset is not performed.
 * nreset and freset are set to 0 if they are not set.
 *
 */
void ResetKineticsSetup::setup()
{
    const auto nScale        = settings::ResetKineticsSettings::getNScale();
    auto       fScale        = settings::ResetKineticsSettings::getFScale();
    const auto nReset        = settings::ResetKineticsSettings::getNReset();
    auto       fReset        = settings::ResetKineticsSettings::getFReset();
    const auto nResetAngular = settings::ResetKineticsSettings::getNResetAngular();
    auto       fResetAngular = settings::ResetKineticsSettings::getFResetAngular();

    const auto numberOfSteps = settings::TimingsSettings::getNumberOfSteps();

    fScale        = (0 == fScale) ? numberOfSteps + 1 : fScale;
    fReset        = (0 == fReset) ? numberOfSteps + 1 : fReset;
    fResetAngular = (0 == fResetAngular) ? numberOfSteps + 1 : fResetAngular;

    _engine.getResetKinetics() = resetKinetics::ResetKinetics(nScale, fScale, nReset, fReset, nResetAngular, fResetAngular);
}