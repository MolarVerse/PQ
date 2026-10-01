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

#ifndef _THERMOSTAT_ENUM_HPP_
#define _THERMOSTAT_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @enum ThermostatType
 *
 * @brief enum class to store the type of thermostat
 *
 */
enum class ThermostatType : std::uint8_t;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define THERMOSTAT_TYPE_LIST(X) \
    X(NONE)                     \
    X(BERENDSEN)                \
    X(VELOCITY_RESCALING)       \
    X(LANGEVIN)                 \
    X(NOSE_HOOVER)

MSTD_ENUM(ThermostatType, std::uint8_t, THERMOSTAT_TYPE_LIST)

#undef THERMOSTAT_TYPE_LIST

namespace mstd
{
    /**
     * @brief Provides a mapping between string aliases and the ThermostatType
     * enum values.
     *
     * @tparam  ThermostatType The enum type for which the input
     * aliases are defined.
     */
    template <>
    struct EnumAliases<ThermostatType>
    {
        static constexpr auto value = makeAliases<ThermostatType>(
            {{"nh_chain", ThermostatType::NOSE_HOOVER},
             {"rescale", ThermostatType::VELOCITY_RESCALING}}
        );
    };

}   // namespace mstd

#endif   // _THERMOSTAT_ENUM_HPP_
