#include "resetKineticsSetup.hpp"

#include "engine.hpp"               // for Engine
#include "resetKinetics.hpp"        // for ResetMomentum, ResetTemperature, resetK...
#include "settings.hpp"             // for Settings
#include "thermostatSettings.hpp"   // for getTargetTemperature
#include "timings.hpp"              // for Timings

using namespace setup;

/**
 * @brief constructs a new Reset Kinetics Setup:: Reset Kinetics Setup object and calls setup
 *
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
    auto nScale = _engine.getSettings().getNScale();
    auto fScale = _engine.getSettings().getFScale();
    auto nReset = _engine.getSettings().getNReset();
    auto fReset = _engine.getSettings().getFReset();

    const auto targetTemperature = settings::ThermostatSettings::getTargetTemperature();

    const auto numberOfSteps = _engine.getTimings().getNumberOfSteps();

    if (nScale != 0 || fScale != 0)
    {
        if (0 == fScale)
            fScale = numberOfSteps + 1;
        if (0 == fReset)
            fReset = numberOfSteps + 1;

        _engine.makeResetKinetics(resetKinetics::ResetTemperature(nScale, fScale, nReset, fReset, targetTemperature));
    }
    else if (nReset != 0 || fReset != 0)
    {
        fScale = numberOfSteps + 1;
        if (0 == fReset)
            fReset = numberOfSteps + 1;

        _engine.makeResetKinetics(resetKinetics::ResetMomentum(nScale, fScale, nReset, fReset, targetTemperature));
    }
    else
        _engine.makeResetKinetics(resetKinetics::ResetKinetics());
}