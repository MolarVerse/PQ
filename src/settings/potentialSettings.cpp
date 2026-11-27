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

#include "potentialSettings.hpp"

#include "enums/potential.hpp"

/**
 * @brief Construct a new Coulomb Potential Config object
 *
 * @param cutoff_ The cutoff distance for the Coulomb potential
 */
CoulombPotentialConfig::CoulombPotentialConfig(const Distance& cutoff_)
    : cutoff(cutoff_),
      cutoffSquared(cutoff_ * cutoff_),
      energyCutoff(1 / cutoff_),
      forceCutoff(1 / (cutoff_ * cutoff_)),
      invCubeCutoff(1 / (cutoff_ * cutoff_ * cutoff_))
{
}

/**
 * @brief Construct a new Potential Settings object
 *
 * Initializes the potential settings with default values.
 */
PotentialSettings::PotentialSettings()
    : _coulombLRType(CoulombLongRangeType::SHIFTED),
      _nonCoulombType(NonCoulombType::GUFF),
      _coulombConfig(defaults::COULOMB_CUT_OFF_DEFAULT)
{
}

/**
 * @brief Finalize the potential settings based on the job type
 *
 * For QM-only jobs, certain potential settings might need to be adjusted,
 * such as setting the Coulomb radius cut off to zero.
 */
void PotentialSettings::finalize(JobType jobType)
{
    if (isQMOnlyJobtype(jobType))
        _coulombConfig = CoulombPotentialConfig(Distance::zero());
}

/**
 * @brief Set the Coulomb radius cut off in the PotentialSettings class
 *
 * @param coulombRadiusCutoff
 */
void PotentialSettings::setCoulombRadiusCutoffByUser(
    const Pos& coulombRadiusCutoff
)
{
    _coulombConfig = CoulombPotentialConfig(coulombRadiusCutoff);
}

/**
 * @brief check if the non-Coulomb radius cut off was set by the user
 *
 * @return true if set by user, false otherwise
 */
bool PotentialSettings::isNonCoulombRadiusCutoffSetByUser() const
{
    return _nonCoulombRadiusCutoff.isSetByUser();
}
