#ifndef _INTRA_NON_BONDED_SETUP_HPP_

#define _INTRA_NON_BONDED_SETUP_HPP_

#include "engine.hpp"

namespace setup
{
    class IntraNonBondedSetup;
    void setupIntraNonBonded(engine::Engine &);
}   // namespace setup

/**
 * @class IntraNonBondedSetup
 *
 * @brief Setup intra non bonded interactions
 *
 */
class setup::IntraNonBondedSetup
{
  private:
    engine::Engine &_engine;

  public:
    explicit IntraNonBondedSetup(engine::Engine &engine) : _engine(engine){};

    void setup();
};

#endif   // _INTRA_NON_BONDED_SETUP_HPP_