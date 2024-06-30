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

#include "manostatSettings.hpp"

#include "stringUtilities.hpp"

using namespace settings;

/**
 * @brief return string of manostatType
 *
 * @param manostatType
 */
std::string settings::string(const ManostatType &manostatType)
{
    switch (manostatType)
    {
        case ManostatType::BERENDSEN: return "berendsen";

        case ManostatType::STOCHASTIC_RESCALING: return "stochastic_rescaling";

        default: return "none";
    }
}

/**
 * @brief return string of isotropy
 *
 * @param isotropy
 */
std::string settings::string(const Isotropy &isotropy)
{
    switch (isotropy)
    {
        using enum Isotropy;

        case ISOTROPIC: return "isotropic";
        case SEMI_ISOTROPIC: return "semi_isotropic";
        case ANISOTROPIC: return "anisotropic";
        case FULL_ANISOTROPIC: return "full_anisotropic";

        default: return "isotropic";
    }
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief sets the manostatType to enum in settings
 *
 * @param manostatType
 */
void ManostatSettings::setManostatType(const std::string_view &manostatType)
{
    using enum ManostatType;
    const auto manostatTypeToLower = utilities::toLowerCopy(manostatType);

    if (manostatTypeToLower == "berendsen")
        _manostatType = BERENDSEN;

    else if (manostatTypeToLower == "stochastic_rescaling")
        _manostatType = STOCHASTIC_RESCALING;

    else
        _manostatType = NONE;
}

/**
 * @brief sets the manostatType to enum in settings
 *
 * @param manostatType
 */
void ManostatSettings::setManostatType(const ManostatType &manostatType)
{
    _manostatType = manostatType;
}

/**
 * @brief sets the isotropy to enum in settings
 *
 * @param isotropy
 */
void ManostatSettings::setIsotropy(const std::string_view &isotropy)
{
    using enum Isotropy;
    const auto isotropyToLower = utilities::toLowerCopy(isotropy);

    if (isotropyToLower == "isotropic")
        _isotropy = ISOTROPIC;

    else if (isotropyToLower == "semi_isotropic")
        _isotropy = SEMI_ISOTROPIC;

    else if (isotropyToLower == "anisotropic")
        _isotropy = ANISOTROPIC;

    else if (isotropyToLower == "full_anisotropic")
        _isotropy = FULL_ANISOTROPIC;

    else
        _isotropy = ISOTROPIC;
}

/**
 * @brief sets the isotropy to enum in settings
 *
 * @param isotropy
 */
void ManostatSettings::setIsotropy(const Isotropy &isotropy)
{
    _isotropy = isotropy;
}

/**
 * @brief sets the pressureSet to bool in settings
 *
 * @param pressureSet
 */
void ManostatSettings::setPressureSet(const bool pressureSet)
{
    _isPressureSet = pressureSet;
}

/**
 * @brief sets the targetPressure to double in settings
 *
 * @param target
 */
void ManostatSettings::setTargetPressure(const double targetPressure)
{
    _targetPressure = targetPressure;
}

/**
 * @brief sets the tauManostat to double in settings
 *
 * @param tauManostat
 */
void ManostatSettings::setTauManostat(const double tauManostat)
{
    _tauManostat = tauManostat;
}

/**
 * @brief sets the compressibility to double in settings
 *
 * @param compressibility
 */
void ManostatSettings::setCompressibility(const double compressibility)
{
    _compressibility = compressibility;
}

/**
 * @brief sets the 2D isotropic axes to vector<size_t> in settings
 *
 * @param indices
 */
void ManostatSettings::set2DIsotropicAxes(const std::vector<size_t> &indices)
{
    _2DIsotropicAxes = indices;
}

/**
 * @brief sets the 2D anisotropic axis to size_t in settings
 *
 * @param index
 */
void ManostatSettings::set2DAnisotropicAxis(const size_t index)
{
    _2DAnisotropicAxis = index;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the manostatType
 *
 * @return ManostatType
 */
ManostatType ManostatSettings::getManostatType() { return _manostatType; }

/**
 * @brief get the isotropy
 *
 * @return Isotropy
 */
Isotropy ManostatSettings::getIsotropy() { return _isotropy; }

/**
 * @brief get if pressure is set
 *
 * @return bool
 */
bool ManostatSettings::isPressureSet() { return _isPressureSet; }

/**
 * @brief get the target pressure
 *
 * @return double
 */
double ManostatSettings::getTargetPressure() { return _targetPressure; }

/**
 * @brief get the tauManostat
 *
 * @return double
 */
double ManostatSettings::getTauManostat() { return _tauManostat; }

/**
 * @brief get the compressibility
 *
 * @return double
 */
double ManostatSettings::getCompressibility() { return _compressibility; }

/**
 * @brief get the 2D isotropic axes
 *
 * @return std::vector<size_t>
 */
std::vector<size_t> ManostatSettings::get2DIsotropicAxes()
{
    return _2DIsotropicAxes;
}

/**
 * @brief get the 2D anisotropic axis
 *
 * @return size_t
 */
size_t ManostatSettings::get2DAnisotropicAxis() { return _2DAnisotropicAxis; }