#include "integratorSetup.hpp"

using namespace std;
using namespace setup;
using namespace engine;

/**
 * @brief wrapper for setupTimings
 *
 */
void setup::setupIntegrator(Engine &engine)
{
    IntegratorSetup integratorSetup(engine);
    integratorSetup.setup();
}

/**
 * @brief sets timestep in integrator
 *
 */
void IntegratorSetup::setup() { _engine.getIntegrator().setDt(_engine.getTimings().getTimestep()); }