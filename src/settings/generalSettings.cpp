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

#include "generalSettings.hpp"

#include <string>
#include <utility>

#include "version.hpp"

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief sets the restart version in the general settings
 *
 * @param restartVersion
 */
void GeneralSettings::setRestartVersionByUser(const SemVer& restartVersion)
{
    _restartVersion.setByUser(restartVersion);
}

/**
 * @brief sets the jobtype to enum in settings
 *
 * @param jobtype
 */
void GeneralSettings::setJobtypeByUser(JobType jobtype)
{
    _jobtype.setByUser(jobtype);
}

/**
 * @brief sets the floating point type
 *
 * @param floatingPointType
 */
void GeneralSettings::setFloatingPointTypeByUser(FPType floatingPointType)
{
    _floatingPointType.setByUser(floatingPointType);
}

/**
 * @brief sets the random seed value
 *
 * @param randomSeed
 */
void GeneralSettings::setRandomSeedByUser(uint_fast32_t randomSeed)
{
    _randomSeed.setByUser(randomSeed);
}

/**
 * @brief sets the integrator type
 *
 * @param integratorType
 */
void GeneralSettings::setIntegratorTypeByUser(IntegratorType integratorType)
{
    _integrator.setByUser(integratorType);
}

/**
 * @brief sets the number of cells for the cell list
 *
 * @param numberOfCells
 */
void GeneralSettings::setNumberOfCellsByUser(size_t numberOfCells)
{
    _numberOfCells.setByUser(numberOfCells);
}

/**
 * @brief gets the number of cells for the cell list
 *
 * @return size_t
 */
size_t GeneralSettings::getNumberOfCells() const
{
    return _numberOfCells.get();
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the restart version
 *
 * @return const SemVer&
 */
const SemVer& GeneralSettings::getRestartVersion() const
{
    return _restartVersion.get();
}

/**
 * @brief get the jobtype
 *
 * @return JobType
 */
JobType GeneralSettings::getJobtype() const { return _jobtype.get(); }

/**
 * @brief checks if the jobtype has been set by the user
 *
 * @return true if set by user, false otherwise
 */
bool GeneralSettings::isJobtypeSetByUser() const
{
    return _jobtype.isSetByUser();
}

/**
 * @brief get the floating point type
 *
 * @return FPType
 */
FPType GeneralSettings::getFloatingPointType() const
{
    return _floatingPointType.get();
}

/**
 * @brief get the floating point string representation used in pybind11
 * bindings
 *
 */
std::string GeneralSettings::getFloatingPointPybindString() const
{
    switch (_floatingPointType.get())
    {
        case FPType::FLOAT: return "float32";
        case FPType::DOUBLE: return "float64";
    }

    std::unreachable();
}

/**
 * @brief get the random seed value
 *
 * @return uint_fast32_t
 */
uint_fast32_t GeneralSettings::getRandomSeed() const
{
    return _randomSeed.get();
}

/**
 * @brief get if the random seed value has been set
 *
 * @return bool
 */
bool GeneralSettings::isRandomSeedSetByUser() const
{
    return _randomSeed.isSetByUser();
}

/**
 * @brief get the integrator type
 *
 * @return IntegratorType
 */
IntegratorType GeneralSettings::getIntegratorType() const
{
    return _integrator.get();
}

/******************************
 *                            *
 * standard is-active methods *
 *                            *
 ******************************/

/**
 * @brief Returns true if the cell list is activated
 *
 * @return true/false
 *
 */
bool GeneralSettings::isCellListActivated() const
{
    return _isCellListActivated.get();
}

/**
 * @brief Returns true if the intra non-bonded interactions are activated
 *
 * @return true/false
 *
 */
bool GeneralSettings::isIntraNonBondedActivated() const
{
    return _isIntraNonBondedActivated.get();
}

/*****************************
 *                           *
 * standard activate methods *
 *                           *
 *****************************/

/**
 * @brief activate cell list
 *
 */
void GeneralSettings::activateCellListByUser()
{
    _isCellListActivated.setByUser(true);
}

/**
 * @brief deactivate cell list
 *
 */
void GeneralSettings::deactivateCellListByUser()
{
    _isCellListActivated.setByUser(false);
}

/**
 * @brief activate intra non-bonded interactions
 *
 */
void GeneralSettings::activateIntraNonBondedByUser()
{
    _isIntraNonBondedActivated.setByUser(true);
}

/**
 * @brief deactivate intra non-bonded interactions
 *
 */
void GeneralSettings::deactivateIntraNonBondedByUser()
{
    _isIntraNonBondedActivated.setByUser(false);
}
