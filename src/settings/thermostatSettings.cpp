/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

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