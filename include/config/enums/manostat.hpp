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

#ifndef _MANOSTAT_ENUM_HPP_
#define _MANOSTAT_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @enum ManostatType
 *
 * @brief enum class to store the type of the manostat
 *
 */
enum class ManostatType : std::uint8_t;

#define MANOSTAT_TYPE_LIST(X) \
    X(NONE)                   \
    X(BERENDSEN)              \
    X(STOCHASTIC_RESCALING)

MSTD_ENUM(ManostatType, std::uint8_t, MANOSTAT_TYPE_LIST);

#undef MANOSTAT_TYPE_LIST

/**
 * @enum Isotropy
 *
 * @brief enum class to store the isotropy of the manostat
 *
 */
enum class Isotropy : std::uint8_t;

#define ISOTROPY_LIST(X) \
    X(NONE)              \
    X(ISOTROPIC)         \
    X(SEMI_ISOTROPIC)    \
    X(ANISOTROPIC)       \
    X(FULL_ANISOTROPIC)

MSTD_ENUM(Isotropy, std::uint8_t, ISOTROPY_LIST);

#undef ISOTROPY_LIST

/**
 * @enum FixedAxis
 *
 * @brief enum class to store the fixed axis of the manostat
 *
 */
enum class FixedAxis : std::uint8_t;

#define U(X) static_cast<unsigned>(X)

#define FIXED_AXIS_LIST(T) \
    T(NONE, 0U)            \
    T(X, 1U << 0U)         \
    T(Y, 1U << 1U)         \
    T(Z, 1U << 2U)         \
    T(XY, U(X) | U(Y))     \
    T(XZ, U(X) | U(Z))     \
    T(YZ, U(Y) | U(Z))     \
    T(ALL, U(X) | U(Y) | U(Z))

MSTD_ENUM_BITFLAG(FixedAxis, std::uint8_t, FIXED_AXIS_LIST);

[[nodiscard]]
bool isAxisFixed(FixedAxis fixedAxis, size_t axisIndex);

#undef FIXED_AXIS_LIST
#undef U

#endif   // _MANOSTAT_ENUM_HPP_
