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

#ifndef _HESSIAN_ENUM_HPP_
#define _HESSIAN_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @brief Enum class for different Hessian builder types.
 *
 */
enum class HessianBuilderType : std::uint8_t;

#define HESSIAN_BUILDER_TYPE_LIST(X) \
    X(CENTRAL)                       \
    X(FORWARD)                       \
    X(FIVE_POINT)                    \
    X(ANALYTIC)

MSTD_ENUM(HessianBuilderType, std::uint8_t, HESSIAN_BUILDER_TYPE_LIST)

#endif   // _HESSIAN_ENUM_HPP_
