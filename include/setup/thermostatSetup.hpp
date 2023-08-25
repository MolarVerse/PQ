#ifndef _THERMOSTAT_SETUP_HPP_

#define _THERMOSTAT_SETUP_HPP_

namespace engine
{
    class Engine;
}   // namespace engine

namespace setup
{
    void setupThermostat(engine::Engine &);

    /**
     * @class ThermostatSetup
     *
     * @brief Setup thermostat
     *
     */
    class ThermostatSetup
    {
      private:
        engine::Engine &_engine;

      public:
        explicit ThermostatSetup(engine::Engine &engine) : _engine(engine){};

        void setup();
    };

}   // namespace setup

#endif   // _THERMOSTAT_SETUP_HPP_