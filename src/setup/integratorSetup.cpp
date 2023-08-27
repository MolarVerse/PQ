#include "integratorSetup.hpp"

#include "engine.hpp"       // for Engine
#include "integrator.hpp"   // for Integrator
#include "timings.hpp"      // for Timings

using namespace setup;

/**
 * @brief constructs a new Integrator Setup:: Integrator Setup object and calls setup
 *
 * @param engine
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