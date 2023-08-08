#include "thermostatSetup.hpp"

#include "constants.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace setup;
using namespace thermostat;
using namespace engine;

/**
 * @brief wrapper for thermostat setup
 *
 * @param engine
 */
void setup::setupThermostat(Engine &engine)
{
    ThermostatSetup thermostatSetup(engine);
    thermostatSetup.setup();
}

/**
 * @brief setup thermostat
 *
 */
void ThermostatSetup::setup()
{
    if (_engine.getSettings().getThermostat() == "berendsen")
    {
        if (!_engine.getSettings().getTemperatureSet())
            throw customException::InputFileException("Temperature not set for Berendsen thermostat");

        _engine.makeThermostat(BerendsenThermostat(_engine.getSettings().getTemperature(),
                                                   _engine.getSettings().getRelaxationTime() * constants::_PS_TO_FS_));
    }

    _engine.getThermostat().setTimestep(_engine.getTimings().getTimestep());
}