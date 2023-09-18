#include "thermostatSettings.hpp"

using settings::ThermostatSettings;

/**
 * @brief return string of thermostatType
 *
 * @param thermostatType
 * @return std::string
 */
std::string settings::string(const ThermostatType &thermostatType)
{
    switch (thermostatType)
    {
    case ThermostatType::BERENDSEN: return "berendsen";
    case ThermostatType::LANGEVIN: return "langevin";
    default: return "none";
    }
}

/**
 * @brief sets the thermostatType to enum in settings
 *
 * @param thermostatType
 */
void ThermostatSettings::setThermostatType(const std::string_view &thermostatType)
{
    if (thermostatType == "berendsen")
        _thermostatType = ThermostatType::BERENDSEN;
    else if (thermostatType == "langevin")
        _thermostatType = ThermostatType::LANGEVIN;
    else
        _thermostatType = ThermostatType::NONE;
}