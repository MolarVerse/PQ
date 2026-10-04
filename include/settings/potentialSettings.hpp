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

#include <mstd/enum.hpp>
#include <optional>

#include "defaults.hpp"
#include "enums/potential.hpp"

namespace settings
{
    /**
     * @class PotentialSettings
     *
     * @brief static class to store settings of the potential
     *
     */
    class PotentialSettings
    {
       private:
        static inline CoulombLongRangeType _coulombLRType =
            CoulombLongRangeType::SHIFTED;
        static inline NonCoulombType _nonCoulombType = NonCoulombType::GUFF;

        static inline double _coulombRadiusCutOff =
            defaults::COULOMB_CUT_OFF_DEFAULT;
        static inline std::optional<double> _nonCoulombRadiusCutOff;
        static inline double                _scale14Coulomb =
            defaults::SCALE_14_COULOMB_DEFAULT;
        static inline double _scale14VanDerWaals =
            defaults::SCALE_14_VAN_DER_WAALS_DEFAULT;

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

        static void setCoulombRadiusCutOff(double coulombRadiusCutOff);
        static void setNonCoulombRadiusCutOff(double nonCoulombRadiusCutOff);
        static void setScale14Coulomb(double scale14Coulomb);
        static void setScale14VanDerWaals(double scale14VanDerWaals);
        static void setReactionFieldEpsilon(double epsilon);
        static void setWolfParameter(double wolfParameter);

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

#endif   // _POTENTIAL_SETTINGS_HPP_
