#include "integratorSetup.hpp"

#include "engine.hpp"       // for Engine
#include "integrator.hpp"   // for Integrator
#include "timings.hpp"      // for Timings

using namespace setup;

/**
 * @brief wrapper for setupTimings
 *
 */
void setup::setupIntegrator(engine::Engine &engine)
{
    IntegratorSetup integratorSetup(engine);
    integratorSetup.setup();
}

/**
 * @brief sets timestep in integrator
 *
 */
void IntegratorSetup::setup() { _engine.getIntegrator().setDt(_engine.getTimings().getTimestep()); }