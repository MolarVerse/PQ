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
#include "enums/jobtype.hpp"
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

    Setting<double, "Scale 1-4 Coulomb"> _scale14Coulomb{
        defaults::SCALE_14_COULOMB_DEFAULT
    };
    Setting<double, "Scale 1-4 Van der Waals"> _scale14VanDerWaals{
        defaults::SCALE_14_VAN_DER_WAALS_DEFAULT
    };

    Setting<InvDistance, "Wolf parameter"> _wolfParameter{
        defaults::WOLF_PARAM_DEFAULT
    };
    Setting<double, "Reaction field epsilon"> _reactionFieldEpsilon{
        defaults::RF_EPSILON_DEFAULT
    };

    CoulombPotentialConfig _coulombConfig;

   public:
    PotentialSettings();
    void finalize(JobType jobType);

    CONST_GETTER(_coulombConfig);

    /********************
     * standard setters *
     ********************/

    void setCoulombLongRangeTypeByUser(CoulombLongRangeType type);
    void setNonCoulombTypeByUser(NonCoulombType type);

    void setCoulombRadiusCutOffByUser(const Pos& coulombRadiusCutOff);
    void setNonCoulombRadiusCutOffByUser(const Pos& nonCoulombRadiusCutOff);

    void setScale14CoulombByUser(double scale14Coulomb);
    void setScale14VanDerWaalsByUser(double scale14VanDerWaals);

    void setReactionFieldEpsilonByUser(double epsilon);
    void setWolfParameterByUser(const InvDistance& wolfParameter);

    /********************
     * standard getters *
     ********************/

    [[nodiscard]] CoulombLongRangeType getCoulombLongRangeType() const;
    [[nodiscard]] NonCoulombType       getNonCoulombType() const;

    [[nodiscard]] const Distance& getCoulombRadiusCutOff() const;
    [[nodiscard]] const Distance& getNonCoulombRadiusCutOff() const;
    [[nodiscard]] bool            isNonCoulombRadiusCutOffSetByUser() const;

    [[nodiscard]] double getScale14Coulomb() const;
    [[nodiscard]] double getScale14VDW() const;

    [[nodiscard]] double             getReactionFieldEpsilon() const;
    [[nodiscard]] const InvDistance& getWolfParameter() const;

    friend struct ::PotentialSettingsUtils;
};

#endif   // SETTINGS__POTENTIAL_SETTINGS_HPP
