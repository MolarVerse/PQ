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

#include "coulombLongRangeInputParser.hpp"

#include <cstddef>   // for size_t, std
#include <format>    // for format

#include "exceptions.hpp"   // for exc::InputFileException, customException
#include "parserUtils.hpp"
#include "potentialSettings.hpp"   // for settings::PotentialSettings
#include "stringUtilities.hpp"     // for toLowerCopy

namespace input
{

    /**
     * @brief Construct a new Input File Parser Coulomb Long Range:: Input File
     * Parser Coulomb Long Range object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) long_range "<string>" 2)
     * wolf_param "<double>"
     */
    CoulombLongRangeInputParser::CoulombLongRangeInputParser()
    {
        addKeyword(
            std::string("long_range"),
            bindMember(
                &CoulombLongRangeInputParser::parseCoulombLongRange,
                this
            ),
            false
        );

        addKeyword(
            std::string("wolf_param"),
            bindMember(&CoulombLongRangeInputParser::parseWolfParameter, this),
            false
        );

        addKeyword(
            std::string("rf_epsilon"),
            bindMember(
                &CoulombLongRangeInputParser::parseReactionFieldEpsilon,
                this
            ),
            false
        );
    }

    /**
     * @brief Parse the coulombic long-range correction used in the simulation
     *
     * @details Possible options are:
     * 1) "none" - no long-range correction is used (default) = shifted
     * potential 2) "reaction_field" - reaction field long-range correction is
     * used 3) "wolf" - wolf long-range correction is used
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throws exc::InputFileException if coulombic long-range
     * correction is not valid - currently only none and wolf are supported
     */
    void CoulombLongRangeInputParser::parseCoulombLongRange(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto type =
            utilities::toLowerAndReplaceDashesCopy(lineElements[2]);

        using enum settings::CoulombLongRangeType;

        if (type == "none" || type == "shifted")
            settings::PotentialSettings::setCoulombLongRangeType(SHIFTED);

        else if (type == "reaction_field")
        {
            settings::PotentialSettings::setCoulombLongRangeType(
                REACTION_FIELD
            );
        }

        else if (type == "wolf")
        {
            settings::PotentialSettings::setCoulombLongRangeType(WOLF);
        }
        else
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid long-range type for coulomb correction "
                    "\"{}\" at line {} in input file\n"
                    "Possible options are: none, shifted, reaction-field, wolf",
                    lineElements[2],
                    lineNumber
                )
            );
        }
    }

    /**
     * @brief parse the wolf parameter used in the simulation
     *
     * @details default value is 0.25
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throws exc::InputFileException if wolf parameter is negative
     */
    void CoulombLongRangeInputParser::parseWolfParameter(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto wolfParameter =
            utilities::stringToFiniteDouble(lineElements[2]);

        if (wolfParameter < 0.0)
            throw exc::InputFileException("Wolf parameter cannot be negative");

        settings::PotentialSettings::setWolfParameter(wolfParameter);
    }

    /**
     * @brief parse the reaction field epsilon used in the simulation
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throws exc::InputFileException if epsilon is negative
     */
    void CoulombLongRangeInputParser::parseReactionFieldEpsilon(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto epsilon = utilities::stringToFiniteDouble(lineElements[2]);

        if (epsilon < 1)
        {
            throw exc::InputFileException(
                "Static relative permittivity \"rf_epsilon\" cannot be "
                "lower than 1.0"
            );
        }

        settings::PotentialSettings::setReactionFieldEpsilon(epsilon);
    }

}   // namespace input
