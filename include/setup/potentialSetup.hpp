#ifndef _POTENTIAL_SETUP_HPP_

#define _POTENTIAL_SETUP_HPP_

#include "engine.hpp"

namespace setup
{
    class PotentialSetup;
    void setupPotential(engine::Engine &);
}   // namespace setup

/**
 * @class PotentialSetup
 *
 * @brief Setup potential
 *
 */
class setup::PotentialSetup
{
  private:
    engine::Engine &_engine;

  public:
    explicit PotentialSetup(engine::Engine &engine) : _engine(engine){};

    void setup();
    void setupCoulomb();
    void setupNonCoulomb();
    void setupNonCoulombicPairs();
};

#endif   // _POTENTIAL_SETUP_HPP_