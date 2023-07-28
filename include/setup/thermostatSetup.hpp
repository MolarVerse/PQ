#ifndef _THERMOSTAT_SETUP_HPP_

#define _THERMOSTAT_SETUP_HPP_

#include "engine.hpp"

namespace setup
{
    class ThermostatSetup;
    void setupThermostat(engine::Engine &);
}   // namespace setup

/**
 * @class ThermostatSetup
 *
 * @brief Setup thermostat
 *
 */
class setup::ThermostatSetup
{
  private:
    engine::Engine &_engine;

  public:
    explicit ThermostatSetup(engine::Engine &engine) : _engine(engine){};

    void setup();
};

#endif