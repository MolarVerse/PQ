#include "resetKineticsSetup.hpp"

using namespace std;
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
        if (fScale == 0) fScale = numberOfSteps + 1;
        if (fReset == 0) fReset = numberOfSteps + 1;
        _engine._resetKinetics = make_unique<ResetTemperature>(nScale, fScale, nReset, fReset, targetTemperature);
    }
    else if (nReset != 0 || fReset != 0)
    {
        fScale = numberOfSteps + 1;
        if (fReset == 0) fReset = numberOfSteps + 1;
        _engine._resetKinetics = make_unique<ResetMomentum>(nScale, fScale, nReset, fReset, targetTemperature);
    }
}