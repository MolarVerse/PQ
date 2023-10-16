/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "defaults.hpp"   // for _COULOMB_LONG_RANGE_TYPE_DEFAULT_, ...

#include <cstddef>       // for size_t
#include <string>        // for allocator, string
#include <string_view>   // for string_view

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

    [[nodiscard]] std::string string(const NonCoulombType nonCoulombType);

    /**
     * @class PotentialSettings
     *
     * @brief static class to store settings of the potential
     *
     */
    class PotentialSettings
    {
      private:
        static inline std::string    _coulombLongRangeType = defaults::_COULOMB_LONG_RANGE_TYPE_DEFAULT_;   // shifted potential
        static inline NonCoulombType _nonCoulombType       = NonCoulombType::GUFF;                          // LJ

        static inline double _coulombRadiusCutOff = defaults::_COULOMB_CUT_OFF_DEFAULT_;   // default is 12.5 Angstrom

        static inline double _scale14Coulomb     = defaults::_SCALE_14_COULOMB_DEFAULT_;         // default is 1.0
        static inline double _scale14VanDerWaals = defaults::_SCALE_14_VAN_DER_WAALS_DEFAULT_;   // default is 1.0

        static inline double _wolfParameter = defaults::_WOLF_PARAMETER_DEFAULT_;   // default is 0.25

      public:
        PotentialSettings()  = default;
        ~PotentialSettings() = default;

        static void setNonCoulombType(const std::string_view &type);
        static void setNonCoulombType(const NonCoulombType type) { _nonCoulombType = type; }

        /********************
         * standard setters *
         ********************/

        static void setCoulombLongRangeType(const std::string_view &type) { _coulombLongRangeType = type; }

        static void setCoulombRadiusCutOff(const double coulombRadiusCutOff) { _coulombRadiusCutOff = coulombRadiusCutOff; }
        static void setScale14Coulomb(const double scale14Coulomb) { _scale14Coulomb = scale14Coulomb; }
        static void setScale14VanDerWaals(const double scale14VanDerWaals) { _scale14VanDerWaals = scale14VanDerWaals; }
        static void setWolfParameter(const double wolfParameter) { _wolfParameter = wolfParameter; }

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static std::string    getCoulombLongRangeType() { return _coulombLongRangeType; }
        [[nodiscard]] static NonCoulombType getNonCoulombType() { return _nonCoulombType; }

        [[nodiscard]] static double getCoulombRadiusCutOff() { return _coulombRadiusCutOff; }
        [[nodiscard]] static double getScale14Coulomb() { return _scale14Coulomb; }
        [[nodiscard]] static double getScale14VanDerWaals() { return _scale14VanDerWaals; }
        [[nodiscard]] static double getWolfParameter() { return _wolfParameter; }
    };

}   // namespace settings

#endif   // _POSITION_SETTINGS_HPP_