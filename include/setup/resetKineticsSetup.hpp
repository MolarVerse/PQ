#ifndef _RESET_KINETICS_SETUP_HPP_

#define _RESET_KINETICS_SETUP_HPP_

#include "engine.hpp"

namespace setup
{
    class ResetKineticsSetup;
    void setupResetKinetics(engine::Engine &);
}   // namespace setup

/**
 * @class ResetKineticsSetup
 *
 * @brief Setup reset kinetics
 *
 */
class setup::ResetKineticsSetup
{
  private:
    engine::Engine &_engine;

  public:
    explicit ResetKineticsSetup(engine::Engine &engine) : _engine(engine){};

    void setup();
};

#endif   // _RESET_KINETICS_SETUP_HPP_