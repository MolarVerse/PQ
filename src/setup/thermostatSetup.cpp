#include "thermostatSetup.hpp"

#include "constants.hpp"    // for _PS_TO_FS_
#include "engine.hpp"       // for Engine
#include "exceptions.hpp"   // for InputFileException
#include "settings.hpp"     // for Settings
#include "thermostat.hpp"   // for BerendsenThermostat, Thermostat, thermostat
#include "timings.hpp"      // for Timings

#include <format>        // for format
#include <string>        // for operator==
#include <string_view>   // for string_view

using namespace setup;

/**
 * @brief wrapper for thermostat setup
 *
 * @details constructs a thermostat setup object and calls the setup function
 *
 * @param engine
 */
void setup::setupThermostat(engine::Engine &engine)
{
    ThermostatSetup thermostatSetup(engine);
    thermostatSetup.setup();
}

/**
 * @brief setup thermostat
 *
 * @details checks if a thermostat was set in the input file,
 * If a thermostat was selected than the user has to provide a temperature for the thermostat.
 *
 * @note the base class Thermostat does not apply any temperature coupling to the system and therefore it represents the none
 * thermostat.
 *
 *
 * @throws InputFileException if no temperature was set for the thermostat
 *
 */
void ThermostatSetup::setup()
{
    const auto thermostat = _engine.getSettings().getThermostat();

    if (thermostat != "none")
        if (!_engine.getSettings().getTemperatureSet())
            throw customException::InputFileException(std::format("Temperature not set for {} thermostat", thermostat));

    if (thermostat == "berendsen")
        _engine.makeThermostat(thermostat::BerendsenThermostat(
            _engine.getSettings().getTemperature(), _engine.getSettings().getRelaxationTime() * constants::_PS_TO_FS_));
    else
        _engine.makeThermostat(thermostat::Thermostat());

    _engine.getThermostat().setTimestep(_engine.getTimings().getTimestep());
}