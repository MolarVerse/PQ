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

#ifndef SETTINGS__POTENTIAL_SETTINGS_HPP
#define SETTINGS__POTENTIAL_SETTINGS_HPP

#include <mstd/enum.hpp>
#include <mstd/utils/members.hpp>

#include "defaults.hpp"
#include "enums/potential.hpp"
#include "setting.hpp"
#include "unit_types.hpp"

/**
 * @struct CoulombPotentialConfig
 *
 * @brief Configuration structure for Coulomb potential parameters
 *
 */
struct CoulombPotentialConfig
{
    Distance           cutoff;
    DistanceSquared    cutoffSquared;
    InvDistance        energyCutoff;
    InvDistanceSquared forceCutoff;
    InvDistanceCubed   invCubeCutoff;

    explicit CoulombPotentialConfig(const Distance& cutoff_);
};

struct PotentialSettingsUtils;

/**
 * @class PotentialSettings
 *
 * @brief static class to store settings of the potential
 *
 */
class PotentialSettings
{
   private:
    Setting<CoulombLongRangeType, "Coulomb long range type"> _coulombLRType;
    Setting<NonCoulombType, "Non coulomb type">              _nonCoulombType;

    Setting<Distance, "Non coulomb radius cut off"> _nonCoulombRadiusCutOff;

    static inline double _scale14Coulomb = defaults::SCALE_14_COULOMB_DEFAULT;
    static inline double _scale14VanDerWaals =
        defaults::SCALE_14_VAN_DER_WAALS_DEFAULT;

    static inline InvDistance _wolfParameter   = defaults::WOLF_PARAM_DEFAULT;
    static inline double _reactionFieldEpsilon = defaults::RF_EPSILON_DEFAULT;

    CoulombPotentialConfig _coulombConfig;

   public:
    PotentialSettings();
    void finalize();

    CONST_GETTER(_coulombConfig);

    /********************
     * standard setters *
     ********************/

    void setCoulombLongRangeTypeByUser(CoulombLongRangeType type);
    void setNonCoulombTypeByUser(NonCoulombType type);

    void setCoulombRadiusCutOffByUser(const Pos& coulombRadiusCutOff);
    void setNonCoulombRadiusCutOffByUser(const Pos& nonCoulombRadiusCutOff);

    static void setScale14Coulomb(double scale14Coulomb);
    static void setScale14VanDerWaals(double scale14VanDerWaals);
    static void setReactionFieldEpsilon(double epsilon);
    static void setWolfParameter(const InvDistance& wolfParameter);

    /********************
     * standard getters *
     ********************/

    [[nodiscard]] CoulombLongRangeType getCoulombLongRangeType() const;
    [[nodiscard]] NonCoulombType       getNonCoulombType() const;

    [[nodiscard]] const Distance&    getCoulombRadiusCutOff() const;
    [[nodiscard]] const Distance&    getNonCoulombRadiusCutOff() const;
    [[nodiscard]] bool               isNonCoulombRadiusCutOffSetByUser() const;
    [[nodiscard]] static double      getScale14Coulomb();
    [[nodiscard]] static double      getScale14VDW();
    [[nodiscard]] static double      getReactionFieldEpsilon();
    [[nodiscard]] static InvDistance getWolfParameter();

    friend struct ::PotentialSettingsUtils;
};

#endif   // SETTINGS__POTENTIAL_SETTINGS_HPP
