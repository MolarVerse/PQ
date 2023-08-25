#include "resetKineticsSetup.hpp"

#include "engine.hpp"          // for Engine
#include "resetKinetics.hpp"   // for ResetMomentum, ResetTemperature, resetK...
#include "settings.hpp"        // for Settings
#include "timings.hpp"         // for Timings

using namespace setup;
using namespace resetKinetics;

/**
 * @brief wrapper for setupResetKinetics
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
 */
void ResetKineticsSetup::setup()
{
    auto nScale = _engine.getSettings().getNScale();
    auto fScale = _engine.getSettings().getFScale();
    auto nReset = _engine.getSettings().getNReset();
    auto fReset = _engine.getSettings().getFReset();

    const auto targetTemperature = _engine.getSettings().getTemperature();

    const auto numberOfSteps = _engine.getTimings().getNumberOfSteps();

    if (nScale != 0 || fScale != 0)
    {
        if (0 == fScale)
            fScale = numberOfSteps + 1;
        if (0 == fReset)
            fReset = numberOfSteps + 1;
        _engine.makeResetKinetics(ResetTemperature(nScale, fScale, nReset, fReset, targetTemperature));
    }
    else if (nReset != 0 || fReset != 0)
    {
        fScale = numberOfSteps + 1;
        if (0 == fReset)
            fReset = numberOfSteps + 1;
        _engine.makeResetKinetics(ResetMomentum(nScale, fScale, nReset, fReset, targetTemperature));
    }
}