#include "integratorSetup.hpp"

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
void IntegratorSetup::setup() {}