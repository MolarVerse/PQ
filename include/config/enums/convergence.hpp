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

#ifndef _CONVERGENCE_ENUM_HPP_
#define _CONVERGENCE_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @enum ConvStrategy
 *
 * @brief Enum representing different types of convergence criteria.
 */
enum class ConvStrategy : std::uint8_t;

#define CONVERGENCE_TYPE_LIST(X) \
    X(RIGOROUS)                  \
    X(LOOSE)                     \
    X(ABSOLUTE)                  \
    X(RELATIVE)

MSTD_ENUM(ConvStrategy, std::uint8_t, CONVERGENCE_TYPE_LIST)

#endif   // _CONVERGENCE_ENUM_HPP_
