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

#ifndef _POTENTIAL_ENUM_HPP_
#define _POTENTIAL_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @brief Enumeration for different types of force fields.
 *
 */
enum class ForceFieldType : std::uint8_t;

#define FF_TYPE_LIST(X) \
    X(OFF)              \
    X(ON)               \
    X(BONDED)

MSTD_ENUM(ForceFieldType, std::uint8_t, FF_TYPE_LIST)

#undef FF_TYPE_LIST

/**
 * @brief Enumeration for different types of non-coulomb interactions.
 *
 */
enum class NonCoulombType : std::uint8_t;

#define NON_COULOMB_TYPE_LIST(X) \
    X(NONE)                      \
    X(LJ)                        \
    X(LJ_9_12)                   \
    X(BUCKINGHAM)                \
    X(MORSE)                     \
    X(GUFF)

MSTD_ENUM(NonCoulombType, std::uint8_t, NON_COULOMB_TYPE_LIST)

/**
 * @brief Input alias for settings::NonCoulombType
 */
template <>
struct mstd::EnumAliases<NonCoulombType>
{
    static constexpr auto value =
        mstd::makeAliases<NonCoulombType>({{"buck", NonCoulombType::BUCKINGHAM}}
        );
};

#undef NON_COULOMB_TYPE_LIST

/**
 * @brief Enumeration for different types of long-range Coulomb interaction
 * methods.
 *
 */
enum class CoulombLongRangeType : std::uint8_t;

#define COULOMB_LONG_RANGE_TYPE_LIST(X) \
    X(SHIFTED)                          \
    X(REACTION_FIELD)                   \
    X(WOLF)

MSTD_ENUM(CoulombLongRangeType, std::uint8_t, COULOMB_LONG_RANGE_TYPE_LIST)

/**
 * @brief Input alias for settings::CoulombLongRangeType
 */
template <>
struct mstd::EnumAliases<CoulombLongRangeType>
{
    static constexpr auto value = mstd::makeAliases<CoulombLongRangeType>(
        {{"none", CoulombLongRangeType::SHIFTED}}
    );
};

#undef COULOMB_LONG_RANGE_TYPE_LIST

#endif   // _POTENTIAL_ENUM_HPP_
