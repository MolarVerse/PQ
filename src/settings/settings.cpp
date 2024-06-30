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

#include "settings.hpp"

#include <string>   // for operator==, string

#include "stringUtilities.hpp"   // for toLowerCopy

using namespace settings;
using namespace utilities;

/**
 * @brief convert jobtype to string representation
 *
 * @param jobtype
 */
std::string settings::string(const JobType jobtype)
{
    switch (jobtype)
    {
        using enum JobType;

        case MM_MD: return "MM_MD";
        case QM_MD: return "QM_MD";
        case QMMM_MD: return "QMMM_MD";
        case RING_POLYMER_QM_MD: return "RING_POLYMER_QM_MD";
        case MM_OPT: return "MM_OPT";
        case NONE: return "NONE";

        default: return "NONE";
    }
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief sets the jobtype to enum in settings
 *
 * @param jobtype
 */
void Settings::setJobtype(const std::string_view jobtype)
{
    using enum JobType;
    const auto jobtypeToLower = toLowerCopy(jobtype);

    if (jobtypeToLower == "mmmd")
        setJobtype(MM_MD);

    else if (jobtypeToLower == "qmmd")
        setJobtype(QM_MD);

    else if (jobtypeToLower == "ring_polymer_qmmd")
        setJobtype(RING_POLYMER_QM_MD);

    else if (jobtypeToLower == "qmmmmd")
        setJobtype(QMMM_MD);

    else if (jobtypeToLower == "mmopt")
        setJobtype(MM_OPT);

    else
        setJobtype(NONE);
}

/**
 * @brief sets the jobtype to enum in settings
 *
 * @param jobtype
 */
void Settings::setJobtype(const JobType jobtype)
{
    _jobtype = jobtype;

    switch (jobtype)
    {
        using enum JobType;

        case MM_OPT:   // fallthrough
        case MM_MD:
        {
            activateMM();
            deactivateQM();
            deactivateRingPolymerMD();
            break;
        }

        case QM_MD:
        {
            deactivateMM();
            activateQM();
            deactivateRingPolymerMD();
            break;
        }

        case RING_POLYMER_QM_MD:
        {
            deactivateMM();
            activateQM();
            activateRingPolymerMD();
            break;
        }

        case QMMM_MD:
        {
            activateMM();
            activateQM();
            deactivateRingPolymerMD();
            break;
        }

        // case NONE: fallthrough
        default:
        {
            deactivateMM();
            deactivateQM();
            deactivateRingPolymerMD();
            break;
        }
    }
}

/**
 * @brief sets the floating point type
 *
 * @param floatingPointType
 */
void Settings::setFloatingPointType(const std::string_view floatingPointType)
{
    using enum FPType;
    const auto floatingPointTypeToLower = toLowerCopy(floatingPointType);

    if (floatingPointTypeToLower == "float")
        setFloatingPointType(FLOAT);

    else
        setFloatingPointType(DOUBLE);
}

/**
 * @brief sets the floating point type
 *
 * @param floatingPointType
 */
void Settings::setFloatingPointType(const FPType floatingPointType)
{
    _floatingPointType = floatingPointType;
}

/**
 * @brief sets MM to active
 *
 * @param dimensionality
 */
void Settings::setIsMMActivated(const bool isMM) { _isMMActivated = isMM; }

/**
 * @brief sets QM to active
 *
 * @param dimensionality
 */
void Settings::setIsQMActivated(const bool isQM) { _isQMActivated = isQM; }

/**
 * @brief sets Ring Polymer MD to active
 *
 * @param dimensionality
 */
void Settings::setIsRingPolymerMDActivated(const bool isRingPolymerMD)
{
    _isRingPolymerMDActivated = isRingPolymerMD;
}

/**
 * @brief sets the dimensionality
 *
 * @param dimensionality
 */
void Settings::setDimensionality(const size_t dimensionality)
{
    _dimensionality = dimensionality;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the jobtype
 *
 * @return JobType
 */
JobType Settings::getJobtype() { return _jobtype; }

/**
 * @brief get the floating point type
 *
 * @return FPType
 */
FPType Settings::getFloatingPointType() { return _floatingPointType; }

/**
 * @brief get the floating point string representation used in pybind11 bindings
 *
 */
std::string Settings::getFloatingPointPybindString()
{
    if (_floatingPointType == FPType::FLOAT)
        return "float32";
    else
        return "float64";
}

/**
 * @brief get the dimensionality
 *
 * @return size_t
 */
size_t Settings::getDimensionality() { return _dimensionality; }

/******************************
 *                            *
 * standard is-active methods *
 *                            *
 ******************************/

/**
 * @brief Returns true if the jobtype does no use any MM type simulations
 *
 * @return true/false if the jobtype does no use any MM type simulations
 *
 */
bool Settings::isQMOnly()
{
    using enum JobType;

    if (_jobtype == QM_MD)
        return true;

    else if (_jobtype == RING_POLYMER_QM_MD)
        return true;

    else
        return false;
}

/**
 * @brief Returns true if the jobtype does is based on MD simulations
 *
 * @return true/false
 *
 */
bool Settings::isMDJobType()
{
    using enum JobType;

    auto isMD = false;
    isMD      = isMD || _jobtype == MM_MD;
    isMD      = isMD || _jobtype == QM_MD;
    isMD      = isMD || _jobtype == QMMM_MD;
    isMD      = isMD || _jobtype == RING_POLYMER_QM_MD;

    return isMD;
}

/**
 * @brief Returns true if the jobtype does is based on optimization
 *
 * @return true/false
 *
 */
bool Settings::isOptJobType() { return _jobtype == JobType::MM_OPT; }

/**
 * @brief Returns true if the MM simulations are activated
 *
 * @return true/false
 *
 */
bool Settings::isMMActivated() { return _isMMActivated; }

/**
 * @brief Returns true if the QM simulations are activated
 *
 * @return true/false
 *
 */
bool Settings::isQMActivated() { return _isQMActivated; }

/**
 * @brief Returns true if both MM and QM simulations are activated
 *
 * @return true/false
 *
 */
bool Settings::isQMMMActivated() { return _isMMActivated && _isQMActivated; }

/**
 * @brief Returns true if only QM simulations are activated
 *
 * @return true/false
 *
 */
bool Settings::isQMOnlyActivated() { return _isQMActivated && !_isMMActivated; }

/**
 * @brief Returns true if only MM simulations are activated
 *
 * @return true/false
 *
 */
bool Settings::isMMOnlyActivated() { return _isMMActivated && !_isQMActivated; }

/**
 * @brief Returns true if the ring polymer MD simulations are activated
 *
 * @return true/false
 *
 */
bool Settings::isRingPolymerMDActivated() { return _isRingPolymerMDActivated; }

/**
 * @brief Returns true if Kokkos is activated
 *
 * @return true/false
 *
 */
bool Settings::useKokkos() { return _useKokkos; }

/*****************************
 *                           *
 * standard activate methods *
 *                           *
 *****************************/

/**
 * @brief activate MM
 *
 */
void Settings::activateMM() { _isMMActivated = true; }

/**
 * @brief activate QM
 *
 */
void Settings::activateQM() { _isQMActivated = true; }

/**
 * @brief activate ring polymer MD simulations
 *
 */
void Settings::activateRingPolymerMD() { _isRingPolymerMDActivated = true; }

/**
 * @brief activate Kokkos
 *
 */
void Settings::activateKokkos() { _useKokkos = true; }

/**
 * @brief deactivate MM
 *
 */
void Settings::deactivateMM() { _isMMActivated = false; }

/**
 * @brief deactivate QM
 *
 */
void Settings::deactivateQM() { _isQMActivated = false; }

/**
 * @brief deactivate ring polymer MD simulations
 *
 */
void Settings::deactivateRingPolymerMD() { _isRingPolymerMDActivated = false; }