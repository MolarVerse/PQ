#ifndef _THERMOSTAT_SETUP_HPP_

#define _THERMOSTAT_SETUP_HPP_

namespace engine
{
    class Engine;   // forward declaration
}

namespace setup
{
    void setupThermostat(engine::Engine &);

    /**
     * @class ThermostatSetup
     *
     * @brief this class setups up the thermostat for the simulation
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