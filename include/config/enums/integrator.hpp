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

#ifndef _INTEGRATOR_ENUM_HPP_
#define _INTEGRATOR_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>
#include <mstd/enum/enum_string.hpp>

/**
 * @brief Enum class for different types of integrators
 */
enum class IntegratorType : std::uint8_t;

#define INTEGRATOR_TYPE_LIST(X) \
    X(NONE)                     \
    X(VELOCITY_VERLET)

MSTD_ENUM(IntegratorType, std::uint8_t, INTEGRATOR_TYPE_LIST)

namespace mstd
{
    /**
     * @brief Provides string aliases for IntegratorType enum values
     */
    template <>
    struct EnumAliases<IntegratorType>
    {
        static constexpr auto value = makeAliases<IntegratorType>(
            {{"v_verlet", IntegratorType::VELOCITY_VERLET}}
        );
    };
}   // namespace mstd

#undef INTEGRATOR_TYPE_LIST

#endif   // _INTEGRATOR_ENUM_HPP_
