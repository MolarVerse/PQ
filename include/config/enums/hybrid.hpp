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

#ifndef _HYBRID_ENUM_HPP_
#define _HYBRID_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @enum SmoothingMethod
 *
 * @brief enum class to store the type of smoothing method
 *
 */
enum class SmoothingMethod : std::uint8_t;

#define SMOOTHING_METHOD_LIST(X) \
    X(HOTSPOT)                   \
    X(EXACT)

MSTD_ENUM(SmoothingMethod, std::uint8_t, SMOOTHING_METHOD_LIST)

#undef SMOOTHING_METHOD_LIST

/**
 * @enum QMForceDist
 *
 * @brief enum class to store the type of force distribution of the QM
 * method in hotspot smoothing
 *
 */
enum class QMForceDist : std::uint8_t;

#define QM_FORCE_DIST_LIST(X) \
    X(NONE)                   \
    X(EQUAL)                  \
    X(RANDOM)                 \
    X(DISTANCE_WEIGHTED)

MSTD_ENUM(QMForceDist, std::uint8_t, QM_FORCE_DIST_LIST)

#undef QM_FORCE_DIST_LIST

#endif   // _HYBRID_ENUM_HPP_
