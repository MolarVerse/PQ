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

#ifndef _POTENTIAL_SETTINGS_HPP_

#define _POTENTIAL_SETTINGS_HPP_

#include <array>
#include <cstdint>
#include <mstd/enum.hpp>
#include <optional>      // for optional
#include <string_view>   // for string_view

#include "defaults.hpp"   // for _COULOMB_LONG_RANGE_TYPE_DEFAULT_, ...
#include "enums/base.hpp"

namespace settings
{

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define FF_TYPE_LIST(X) \
    X(OFF)              \
    X(ON)               \
    X(BONDED)

    MSTD_ENUM(ForceFieldType, std::uint8_t, FF_TYPE_LIST)

#undef FF_TYPE_LIST

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define NON_COULOMB_TYPE_LIST(X) \
    X(NONE)                      \
    X(LJ)                        \
    X(LJ_9_12)                   \
    X(BUCKINGHAM)                \
    X(MORSE)                     \
    X(GUFF)

    MSTD_ENUM(NonCoulombType, std::uint8_t, NON_COULOMB_TYPE_LIST)

#undef NON_COULOMB_TYPE_LIST

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define COULOMB_LONG_RANGE_TYPE_LIST(X) \
    X(SHIFTED)                          \
    X(REACTION_FIELD)                   \
    X(WOLF)

    MSTD_ENUM(CoulombLongRangeType, std::uint8_t, COULOMB_LONG_RANGE_TYPE_LIST)

#undef COULOMB_LONG_RANGE_TYPE_LIST

    /**
     * @class PotentialSettings
     *
     * @brief static class to store settings of the potential
     *
     */
    class PotentialSettings
    {
       private:
        // clang-format off
        static inline CoulombLongRangeType _coulombLRType  = CoulombLongRangeType::SHIFTED;
        static inline NonCoulombType       _nonCoulombType = NonCoulombType::GUFF;

        static inline double _coulombRadiusCutOff = defaults::COULOMB_CUT_OFF_DEFAULT;
        static inline std::optional<double> _nonCoulombRadiusCutOff;
        static inline double _scale14Coulomb      = defaults::SCALE_14_COULOMB_DEFAULT;
        static inline double _scale14VanDerWaals  = defaults::SCALE_14_VAN_DER_WAALS_DEFAULT;
        // clang-format on

        static inline double _wolfParameter = defaults::WOLF_PARAM_DEFAULT;
        static inline double _reactionFieldEpsilon =
            defaults::RF_EPSILON_DEFAULT;

       public:
        PotentialSettings()  = default;
        ~PotentialSettings() = default;

        /********************
         * standard setters *
         ********************/

        static void setNonCoulombType(NonCoulombType type);
        static void setCoulombLongRangeType(CoulombLongRangeType type);

        // clang-format off
        static void setCoulombRadiusCutOff(double coulombRadiusCutOff);
        static void setNonCoulombRadiusCutOff(double nonCoulombRadiusCutOff);
        static void setScale14Coulomb(double scale14Coulomb);
        static void setScale14VanDerWaals(double scale14VanDerWaals);
        static void setReactionFieldEpsilon(double epsilon);
        static void setWolfParameter(double wolfParameter);
        // clang-format on

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static CoulombLongRangeType getCoulombLongRangeType();
        [[nodiscard]] static NonCoulombType       getNonCoulombType();

        [[nodiscard]] static double                getCoulombRadiusCutOff();
        [[nodiscard]] static std::optional<double> getNonCoulombRadiusCutOff();
        [[nodiscard]] static double                getScale14Coulomb();
        [[nodiscard]] static double                getScale14VDW();
        [[nodiscard]] static double                getReactionFieldEpsilon();
        [[nodiscard]] static double                getWolfParameter();
    };

}   // namespace settings

// TODO: move this to deidcated enum file as soon as it is done

/**
 * @brief Input alias for settings::CoulombLongRangeType
 */
template <>
struct InputAlias<settings::CoulombLongRangeType>
{
    static constexpr std::
        array<std::pair<std::string_view, settings::CoulombLongRangeType>, 1>
            value = {{{"none", settings::CoulombLongRangeType::SHIFTED}}};
};

/**
 * @brief Input alias for settings::NonCoulombType
 */
template <>
struct InputAlias<settings::NonCoulombType>
{
    static constexpr std::
        array<std::pair<std::string_view, settings::NonCoulombType>, 1>
            value = {{{"buck", settings::NonCoulombType::BUCKINGHAM}}};
};

#endif   // _POTENTIAL_SETTINGS_HPP_
