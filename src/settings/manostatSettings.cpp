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

#include "manostatSettings.hpp"

#include "stringUtilities.hpp"

using settings::ManostatSettings;

/**
 * @brief return string of manostatType
 *
 * @param manostatType
 */
std::string settings::string(const settings::ManostatType &manostatType)
{
    switch (manostatType)
    {
    case settings::ManostatType::BERENDSEN: return "berendsen";

    case settings::ManostatType::STOCHASTIC_RESCALING: return "stochastic_rescaling";

    default: return "none";
    }
}

/**
 * @brief sets the manostatType to enum in settings
 *
 * @param manostatType
 */
void ManostatSettings::setManostatType(const std::string_view &manostatType)
{
    const auto manostatTypeToLower = utilities::toLowerCopy(manostatType);

    if (manostatTypeToLower == "berendsen")
        _manostatType = settings::ManostatType::BERENDSEN;

    else if (manostatTypeToLower == "stochastic_rescaling")
        _manostatType = settings::ManostatType::STOCHASTIC_RESCALING;

    else
        _manostatType = settings::ManostatType::NONE;
}

/**
 * @brief return string of isotropy
 *
 * @param isotropy
 */
std::string settings::string(const settings::Isotropy &isotropy)
{
    switch (isotropy)
    {
    case settings::Isotropy::ISOTROPIC: return "isotropic";

    case settings::Isotropy::SEMI_ISOTROPIC: return "semi_isotropic";

    case settings::Isotropy::ANISOTROPIC: return "anisotropic";

    default: return "isotropic";
    }
}

/**
 * @brief sets the isotropy to enum in settings
 *
 * @param isotropy
 */
void ManostatSettings::setIsotropy(const std::string_view &isotropy)
{
    const auto isotropyToLower = utilities::toLowerCopy(isotropy);

    if (isotropyToLower == "isotropic")
        _isotropy = settings::Isotropy::ISOTROPIC;

    else if (isotropyToLower == "semi_isotropic")
        _isotropy = settings::Isotropy::SEMI_ISOTROPIC;

    else if (isotropyToLower == "anisotropic")
        _isotropy = settings::Isotropy::ANISOTROPIC;

    else
        _isotropy = settings::Isotropy::ISOTROPIC;
}