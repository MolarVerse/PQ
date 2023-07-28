#ifndef _INTEGRATOR_SETUP_HPP_

#define _INTEGRATOR_SETUP_HPP_

#include "engine.hpp"

namespace setup
{
    class IntegratorSetup;
    void setupIntegrator(engine::Engine &);
}   // namespace setup

/**
 * @class IntegratorSetup
 *
 * @brief Setup Integrator
 *
 */
class setup::IntegratorSetup
{
  private:
    engine::Engine &_engine;

  public:
    explicit IntegratorSetup(engine::Engine &engine) : _engine(engine){};

    void setup();
};

#endif   // _INTEGRATOR_SETUP_HPP_