#include "thermostatSetup.hpp"

#include "constants.hpp"    // for _PS_TO_FS_
#include "engine.hpp"       // for Engine
#include "exceptions.hpp"   // for InputFileException
#include "settings.hpp"     // for Settings
#include "thermostat.hpp"   // for BerendsenThermostat, Thermostat, thermostat
#include "timings.hpp"      // for Timings

#include <string>   // for operator==

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