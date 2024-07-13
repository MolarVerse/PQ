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

#include "exceptions.hpp"
#include "stringUtilities.hpp"

using namespace settings;
using namespace utilities;
using namespace customException;

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
        using enum NonCoulombType;

        case LJ: return "lj";
        case LJ_9_12: return "lj_9_12";
        case BUCKINGHAM: return "buck";
        case MORSE: return "morse";
        case GUFF: return "guff";

        default: return "none";
    }
}

/**
 * @brief return string of CoulombLongRangeType
 *
 * @param coulombLongRangeType
 * @return std::string
 */
std::string settings::string(const CoulombLongRangeType coulombLongRangeType)
{
    switch (coulombLongRangeType)
    {
        using enum CoulombLongRangeType;

        case WOLF: return "wolf";
        case SHIFTED: return "shifted";

        default: return "shifted";
    }
}

/********************
 *                  *
 * standard setters *
 *                  *
 ********************/

/**
 * @brief Set the nonCoulomb type as string and enum in the PotentialSettings
 * class
 *
 * @param type
 */
void PotentialSettings::setNonCoulombType(const std::string_view &type)
{
    using enum NonCoulombType;
    const auto typeToLower = toLowerCopy(type);

    if (typeToLower == "lj")
        _nonCoulombType = LJ;

    else if (typeToLower == "lj_9_12")
        _nonCoulombType = LJ_9_12;

    else if (typeToLower == "buck")
        _nonCoulombType = BUCKINGHAM;

    else if (typeToLower == "morse")
        _nonCoulombType = MORSE;

    else if (typeToLower == "guff")
        _nonCoulombType = GUFF;

    else
        _nonCoulombType = NONE;
}

/**
 * @brief Set the nonCoulomb type as enum in the PotentialSettings class
 *
 * @param type
 */
void PotentialSettings::setNonCoulombType(const NonCoulombType type)
{
    _nonCoulombType = type;
}

void PotentialSettings::setCoulombLongRangeType(const std::string_view &type)
{
    using enum CoulombLongRangeType;
    const auto typeToLower = toLowerCopy(type);

    if (typeToLower == "wolf")
        _coulombLRType = WOLF;

    else if (typeToLower == "shifted")
        _coulombLRType = SHIFTED;

    else
        throw UserInputException(
            "Unknown Coulomb long range type " + std::string(type)
        );
}

/**
 * @brief Set the Coulomb long range type in the PotentialSettings class
 *
 * @param type
 */
void PotentialSettings::setCoulombLongRangeType(const CoulombLongRangeType &type
)
{
    _coulombLRType = type;
}

/**
 * @brief Set the Coulomb radius cut off in the PotentialSettings class
 *
 * @param coulombRadiusCutOff
 */
void PotentialSettings::setCoulombRadiusCutOff(const double coulombRadiusCutOff)
{
    _coulombRadiusCutOff = coulombRadiusCutOff;
}

/**
 * @brief Set the 1-4 Coulomb scaling factor in the PotentialSettings class
 *
 * @param scale14Coulomb
 */
void PotentialSettings::setScale14Coulomb(const double scale14Coulomb)
{
    _scale14Coulomb = scale14Coulomb;
}

/**
 * @brief Set the 1-4 Van der Waals scaling factor in the PotentialSettings
 * class
 *
 * @param scale14VanDerWaals
 */
void PotentialSettings::setScale14VanDerWaals(const double scale14VanDerWaals)
{
    _scale14VanDerWaals = scale14VanDerWaals;
}

/**
 * @brief Set the Wolf parameter in the PotentialSettings class
 *
 * @param wolfParameter
 */
void PotentialSettings::setWolfParameter(const double wolfParameter)
{
    _wolfParameter = wolfParameter;
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
CoulombLongRangeType PotentialSettings::getCoulombLongRangeType()
{
    return _coulombLRType;
}

/**
 * @brief get the nonCoulomb type
 *
 * @return NonCoulombType
 */
NonCoulombType PotentialSettings::getNonCoulombType()
{
    return _nonCoulombType;
}

/**
 * @brief get the Coulomb radius cut off
 *
 * @return double
 */
double PotentialSettings::getCoulombRadiusCutOff()
{
    return _coulombRadiusCutOff;
}

/**
 * @brief get the 1-4 Coulomb scaling factor
 *
 * @return double
 */
double PotentialSettings::getScale14Coulomb() { return _scale14Coulomb; }

/**
 * @brief get the 1-4 Van der Waals scaling factor
 *
 * @return double
 */
double PotentialSettings::getScale14VDW() { return _scale14VanDerWaals; }

/**
 * @brief get the Wolf parameter
 *
 * @return double
 */
double PotentialSettings::getWolfParameter() { return _wolfParameter; }