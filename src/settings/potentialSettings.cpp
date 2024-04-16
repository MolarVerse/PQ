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

#include "stringUtilities.hpp"

using namespace settings;

/**
 * @brief return string of nonCoulombType
 *
 * @param nonCoulombType
 * @return std::string
 */
std::string settings::string(const NonCoulombType nonCoulombType)
{
    switch (nonCoulombType)
    {
    case NonCoulombType::LJ: return "lj";
    case NonCoulombType::LJ_9_12: return "lj_9_12";
    case NonCoulombType::BUCKINGHAM: return "buck";
    case NonCoulombType::MORSE: return "morse";
    case NonCoulombType::GUFF: return "guff";
    default: return "none";
    }
}

/**
 * @brief Set the nonCoulomb type as string and enum in the PotentialSettings class
 *
 * @param type
 */
void PotentialSettings::setNonCoulombType(const std::string_view &type)
{
    const auto typeToLower = utilities::toLowerCopy(type);

    if (typeToLower == "lj")
        _nonCoulombType = NonCoulombType::LJ;
    else if (typeToLower == "lj_9_12")
        _nonCoulombType = NonCoulombType::LJ_9_12;
    else if (typeToLower == "buck")
        _nonCoulombType = NonCoulombType::BUCKINGHAM;
    else if (typeToLower == "morse")
        _nonCoulombType = NonCoulombType::MORSE;
    else if (typeToLower == "guff")
        _nonCoulombType = NonCoulombType::GUFF;
    else
        _nonCoulombType = NonCoulombType::NONE;
}