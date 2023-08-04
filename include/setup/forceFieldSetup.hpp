#ifndef _FORCE_FIELD_SETUP_HPP_

#define _FORCE_FIELD_SETUP_HPP_

#include "engine.hpp"

namespace setup
{
    class ForceFieldSetup;
    void setupForceField(engine::Engine &);
}   // namespace setup

/**
 * @class SetupCellList
 *
 */
class setup::ForceFieldSetup
{
  private:
    engine::Engine &_engine;

  public:
    explicit ForceFieldSetup(engine::Engine &engine) : _engine(engine){};

    void setup(){};
};

#endif   // _FORCE_FIELD_SETUP_HPP_