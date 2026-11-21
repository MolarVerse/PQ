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

/********************
 *                  *
 * standard setters *
 *                  *
 ********************/

/**
 * @brief Set the nonCoulomb type as enum in the PotentialSettings class
 *
 * @param type
 */
void PotentialSettings::setNonCoulombTypeByUser(NonCoulombType type)
{
    _nonCoulombType.setByUser(type);
}

/**
 * @brief Set the Coulomb long range type in the PotentialSettings class
 *
 * @param type
 */
void PotentialSettings::setCoulombLongRangeTypeByUser(CoulombLongRangeType type)
{
    _coulombLRType.setByUser(type);
}

/**
 * @brief Set the Coulomb radius cut off in the PotentialSettings class
 *
 * @param coulombRadiusCutOff
 */
void PotentialSettings::setCoulombRadiusCutOffByUser(
    const Pos& coulombRadiusCutOff
)
{
    _coulombConfig = CoulombPotentialConfig(coulombRadiusCutOff);
}

/**
 * @brief Set the non-Coulomb radius cut off in the PotentialSettings class
 *
 * @param nonCoulombRadiusCutOff
 */
void PotentialSettings::setNonCoulombRadiusCutOffByUser(
    const Pos& nonCoulombRadiusCutOff
)
{
    _nonCoulombRadiusCutOff.setByUser(nonCoulombRadiusCutOff);
}

/**
 * @brief Set the 1-4 Coulomb scaling factor in the PotentialSettings class
 *
 * @param scale14Coulomb
 */
void PotentialSettings::setScale14CoulombByUser(double scale14Coulomb)
{
    _scale14Coulomb.setByUser(scale14Coulomb);
}

/**
 * @brief Set the 1-4 Van der Waals scaling factor in the PotentialSettings
 * class
 *
 * @param scale14VanDerWaals
 */
void PotentialSettings::setScale14VanDerWaalsByUser(double scale14VanDerWaals)
{
    _scale14VanDerWaals.setByUser(scale14VanDerWaals);
}

/**
 * @brief Set the reaction field epsilon in the PotentialSettings class
 *
 * @param epsilon
 */
void PotentialSettings::setReactionFieldEpsilonByUser(double epsilon)
{
    _reactionFieldEpsilon.setByUser(epsilon);
}

/**
 * @brief Set the Wolf parameter in the PotentialSettings class
 *
 * @param wolfParameter
 */
void PotentialSettings::setWolfParameterByUser(const InvDistance& wolfParameter)
{
    _wolfParameter.setByUser(wolfParameter);
}

/********************
 *                  *
 * standard getters *
 *                  *
 ********************/

/**
 * @brief get the Coulomb long range type
 *
 * @return CoulombLongRangeType
 */
CoulombLongRangeType PotentialSettings::getCoulombLongRangeType() const
{
    return _coulombLRType.get();
}

/**
 * @brief get the nonCoulomb type
 *
 * @return NonCoulombType
 */
NonCoulombType PotentialSettings::getNonCoulombType() const
{
    return _nonCoulombType.get();
}

/**
 * @brief get the Coulomb radius cut off
 *
 * @return Pos
 */
const Distance& PotentialSettings::getCoulombRadiusCutOff() const
{
    return _coulombConfig.cutoff;
}

/**
 * @brief get the non-Coulomb radius cut off
 *
 * @return std::optional<Pos>
 */
const Distance& PotentialSettings::getNonCoulombRadiusCutOff() const
{
    return _nonCoulombRadiusCutOff.get();
}

/**
 * @brief check if the non-Coulomb radius cut off was set by the user
 *
 * @return true if set by user, false otherwise
 */
bool PotentialSettings::isNonCoulombRadiusCutOffSetByUser() const
{
    return _nonCoulombRadiusCutOff.isSetByUser();
}

/**
 * @brief get the 1-4 Coulomb scaling factor
 *
 * @return double
 */
double PotentialSettings::getScale14Coulomb() const
{
    return _scale14Coulomb.get();
}

/**
 * @brief get the 1-4 Van der Waals scaling factor
 *
 * @return double
 */
double PotentialSettings::getScale14VDW() const
{
    return _scale14VanDerWaals.get();
}

/**
 * @brief get the reaction field epsilon
 *
 * @return double
 */
double PotentialSettings::getReactionFieldEpsilon() const
{
    return _reactionFieldEpsilon.get();
}

/**
 * @brief get the Wolf parameter
 *
 * @return const InvDistance&
 */
const InvDistance& PotentialSettings::getWolfParameter() const
{
    return _wolfParameter.get();
}
