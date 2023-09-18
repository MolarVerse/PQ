#include "thermostatSettings.hpp"

#include "stringUtilities.hpp"   // for toLowerCopy

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
    case ThermostatType::VELOCITY_RESCALING: return "velocity_rescaling";
    case ThermostatType::LANGEVIN: return "langevin";
    case ThermostatType::NOSE_HOOVER: return "nh-chain";
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
    const auto thermostatTypeToLower = utilities::toLowerCopy(thermostatType);

    if (thermostatTypeToLower == "berendsen")
        _thermostatType = ThermostatType::BERENDSEN;
    else if (thermostatTypeToLower == "velocity_rescaling")
        _thermostatType = ThermostatType::VELOCITY_RESCALING;
    else if (thermostatTypeToLower == "langevin")
        _thermostatType = ThermostatType::LANGEVIN;
    else if (thermostatTypeToLower == "nh-chain")
        _thermostatType = ThermostatType::NOSE_HOOVER;
    else
        _thermostatType = ThermostatType::NONE;
}