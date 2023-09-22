#include "resetKineticsSetup.hpp"

#include "engine.hpp"                  // for Engine
#include "resetKinetics.hpp"           // for ResetMomentum, ResetTemperature, resetK...
#include "resetKineticsSettings.hpp"   // for ResetKineticsSettings
#include "thermostatSettings.hpp"      // for getTargetTemperature
#include "timingsSettings.hpp"         // for TimingsSettings

using namespace setup;

/**
 * @brief constructs a new Reset Kinetics Setup:: Reset Kinetics Setup object and calls setup
 *
 * @param engine
 */
void setup::setupResetKinetics(engine::Engine &engine)
{
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

    if ((nScale != 0) || (fScale != 0 && fScale < numberOfSteps + 1))
        _engine.makeResetKinetics(resetKinetics::ResetTemperature(nScale, fScale, nReset, fReset, nResetAngular, fResetAngular));
    else if ((nReset != 0) || (fReset != 0 && fReset < numberOfSteps + 1))
        _engine.makeResetKinetics(resetKinetics::ResetMomentum(nScale, fScale, nReset, fReset, nResetAngular, fResetAngular));
    else
        _engine.makeResetKinetics(resetKinetics::ResetKinetics(nResetAngular, fResetAngular));
}