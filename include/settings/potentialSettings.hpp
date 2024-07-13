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

#ifndef _POSITION_SETTINGS_HPP_

#define _POSITION_SETTINGS_HPP_

#include <cstddef>       // for size_t
#include <string>        // for allocator, string
#include <string_view>   // for string_view

#include "defaults.hpp"   // for _COULOMB_LONG_RANGE_TYPE_DEFAULT_, ...

namespace settings
{
    /**
     * @enum CoulombLongRangeType
     *
     * @brief enum class to store the coulomb long range type
     *
     */
    enum class NonCoulombType : size_t
    {
        LJ,
        LJ_9_12,   // at the momentum just dummy for testing not implemented yet
        BUCKINGHAM,
        MORSE,
        GUFF,
        NONE
    };

    /**
     * @enum CoulombLongRangeType
     *
     * @brief enum class to store the coulomb long range type
     *
     */
    enum class CoulombLongRangeType : size_t
    {
        WOLF,
        SHIFTED
    };

    // TODO: implement long range type as enum

    [[nodiscard]] std::string string(const NonCoulombType nonCoulombType);
    [[nodiscard]] std::string string(const CoulombLongRangeType nonCoulombType);

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

        static inline double _coulombRadiusCutOff = defaults::_COULOMB_CUT_OFF_DEFAULT_;
        static inline double _scale14Coulomb      = defaults::_SCALE_14_COULOMB_DEFAULT_;
        static inline double _scale14VanDerWaals  = defaults::_SCALE_14_VAN_DER_WAALS_DEFAULT_;
        // clang-format on

        static inline double _wolfParameter = defaults::_WOLF_PARAM_DEFAULT_;

       public:
        PotentialSettings()  = default;
        ~PotentialSettings() = default;

        /********************
         * standard setters *
         ********************/

        static void setNonCoulombType(const std::string_view &type);
        static void setNonCoulombType(const NonCoulombType type);
        static void setCoulombLongRangeType(const std::string_view &type);
        static void setCoulombLongRangeType(const CoulombLongRangeType &type);

        static void setCoulombRadiusCutOff(const double coulombRadiusCutOff);
        static void setScale14Coulomb(const double scale14Coulomb);
        static void setScale14VanDerWaals(const double scale14VanDerWaals);
        static void setWolfParameter(const double wolfParameter);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static CoulombLongRangeType getCoulombLongRangeType();
        [[nodiscard]] static NonCoulombType       getNonCoulombType();

        [[nodiscard]] static double getCoulombRadiusCutOff();
        [[nodiscard]] static double getScale14Coulomb();
        [[nodiscard]] static double getScale14VDW();
        [[nodiscard]] static double getWolfParameter();
    };

}   // namespace settings

#endif   // _POSITION_SETTINGS_HPP_