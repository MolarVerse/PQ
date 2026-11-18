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

#ifndef CONFIG__ENUMS__OUTPUT_HPP
#define CONFIG__ENUMS__OUTPUT_HPP

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @brief file format of the MD trajectory (positions, velocities, forces and
 * charges)
 */
enum class TrajectoryFormat : std::uint8_t;

#define TRAJECTORY_FORMAT_LIST(X) \
    X(XYZ)                        \
    X(EXTXYZ)

MSTD_ENUM(TrajectoryFormat, std::uint8_t, TRAJECTORY_FORMAT_LIST)

#undef TRAJECTORY_FORMAT_LIST

#endif   // CONFIG__ENUMS__OUTPUT_HPP
