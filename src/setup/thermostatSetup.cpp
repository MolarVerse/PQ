#include "thermostatSetup.hpp"

#include "berendsenThermostat.hpp"   // for BerendsenThermostat
#include "constants.hpp"             // for _PS_TO_FS_
#include "engine.hpp"                // for Engine
#include "exceptions.hpp"            // for InputFileException
#include "thermostat.hpp"            // for BerendsenThermostat, Thermostat, thermostat
#include "thermostatSettings.hpp"    // for ThermostatSettings
#include "timingsSettings.hpp"       // for TimingsSettings

#include <format>   // for format
#include <string>   // for operator==

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
 * If a thermostat was selected than the user has to provide a target temperature for the thermostat.
 *
 * @note the base class Thermostat does not apply any temperature coupling to the system and therefore it represents the none
 * thermostat.
 *
 * @throws InputFileException if no temperature was set for the thermostat
 *
 */
void ThermostatSetup::setup()
{
    const auto thermostatType = settings::ThermostatSettings::getThermostatType();

    if (thermostatType != "none")
        if (!settings::ThermostatSettings::isTemperatureSet())
            throw customException::InputFileException(std::format("Temperature not set for {} thermostat", thermostatType));

    if (thermostatType == "berendsen")
        _engine.makeThermostat(
            thermostat::BerendsenThermostat(settings::ThermostatSettings::getTargetTemperature(),
                                            settings::ThermostatSettings::getRelaxationTime() * constants::_PS_TO_FS_));
    else
        _engine.makeThermostat(thermostat::Thermostat());
}