#ifndef _MANOSTAT_SETUP_HPP_

#define _MANOSTAT_SETUP_HPP_

#include "engine.hpp"

namespace setup
{
    class ManostatSetup;
    void setupManostat(engine::Engine &);
}   // namespace setup

/**
 * @class ManostatSetup
 *
 * @brief Setup manostat
 *
 */
class setup::ManostatSetup
{
  private:
    engine::Engine &_engine;

  public:
    explicit ManostatSetup(engine::Engine &engine) : _engine(engine){};

    void setup();
};

#endif   // _MANOSTAT_SETUP_HPP_