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

    Setting<Distance, "Non coulomb radius cut off"> _nonCoulombRadiusCutoff;

    Setting<double, "Scale 1-4 Coulomb"> _scale14Coulomb{
        defaults::SCALE_14_COULOMB_DEFAULT
    };
    Setting<double, "Scale 1-4 Van der Waals"> _scale14VDW{
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
    SETTING_GETTER_AND_SETTER_BY_USER(_coulombLRType);
    SETTING_GETTER_AND_SETTER_BY_USER(_nonCoulombType);
    SETTING_GETTER_AND_SETTER_BY_USER(_nonCoulombRadiusCutoff);
    SETTING_GETTER_AND_SETTER_BY_USER(_scale14Coulomb);
    SETTING_GETTER_AND_SETTER_BY_USER(_scale14VDW);
    SETTING_GETTER_AND_SETTER_BY_USER(_wolfParameter);
    SETTING_GETTER_AND_SETTER_BY_USER(_reactionFieldEpsilon);

    void setCoulombRadiusCutoffByUser(const Pos& coulombRadiusCutoff);

    [[nodiscard]] const Distance& get_coulombRadiusCutoff() const;
    [[nodiscard]] bool            isNonCoulombRadiusCutoffSetByUser() const;

    friend struct ::PotentialSettingsUtils;
};

/**
 * @brief get the Coulomb radius cut off
 *
 * @return const Distance&
 */
inline const Distance& PotentialSettings::get_coulombRadiusCutoff() const
{
    return _coulombConfig.cutoff;
}

#endif   // SETTINGS__POTENTIAL_SETTINGS_HPP
