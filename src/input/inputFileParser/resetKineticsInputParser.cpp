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

#include "resetKineticsInputParser.hpp"

#include <cstddef>       // for size_t, std
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

#include "exceptions.hpp"   // for InputFileException, customException
#include "resetKineticsSettings.hpp"   // for ResetKineticsSettings

using namespace input;
using namespace engine;
using namespace customException;
using namespace settings;

/**
 * @brief Construct a new Input File Parser Reset Kinetics:: Input File Parser
 * Reset Kinetics object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) nscale <size_t> 2) fscale
 * <size_t> 3) nreset <size_t> 4) freset <size_t>
 *
 * @param engine
 */
ResetKineticsInputParser::ResetKineticsInputParser(Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        std::string("nscale"),
        bind_front(&ResetKineticsInputParser::parseNScale, this),
        false
    );
    addKeyword(
        std::string("fscale"),
        bind_front(&ResetKineticsInputParser::parseFScale, this),
        false
    );
    addKeyword(
        std::string("nreset"),
        bind_front(&ResetKineticsInputParser::parseNReset, this),
        false
    );
    addKeyword(
        std::string("freset"),
        bind_front(&ResetKineticsInputParser::parseFReset, this),
        false
    );
    addKeyword(
        std::string("nreset_angular"),
        bind_front(&ResetKineticsInputParser::parseNResetAngular, this),
        false
    );
    addKeyword(
        std::string("freset_angular"),
        bind_front(&ResetKineticsInputParser::parseFResetAngular, this),
        false
    );
    addKeyword(
        std::string("freset_forces"),
        bind_front(&ResetKineticsInputParser::parseFResetForces, this),
        false
    );
}

/**
 * @brief parse nscale and set it in settings
 *
 * @details default value is 0
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throw InputFileException if nscale is negative
 */
void ResetKineticsInputParser::parseNScale(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto nScale = stoi(lineElements[2]);

    if (nScale < 0)
        throw InputFileException("Nscale must be positive");

    ResetKineticsSettings::setNScale(size_t(nScale));
}

/**
 * @brief parse fscale and set it in settings
 *
 * @details default value is 0 but then set to UINT_MAX in setup
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throw InputFileException if fscale is negative
 */
void ResetKineticsInputParser::parseFScale(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto fScale = stoi(lineElements[2]);

    if (fScale < 0)
        throw InputFileException("Fscale must be positive");

    ResetKineticsSettings::setFScale(static_cast<size_t>(fScale));
}

/**
 * @brief parse nreset and set it in settings
 *
 * @details default value is 0
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throw InputFileException if nreset is negative
 */
void ResetKineticsInputParser::parseNReset(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto nReset = stoi(lineElements[2]);

    if (nReset < 0)
        throw InputFileException("Nreset must be positive");

    ResetKineticsSettings::setNReset(static_cast<size_t>(nReset));
}

/**
 * @brief parse freset and set it in settings
 *
 * @details default value is 0 but then set to UINT_MAX in setup
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throw InputFileException if freset is negative
 */
void ResetKineticsInputParser::parseFReset(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto fReset = stoi(lineElements[2]);

    if (fReset < 0)
        throw InputFileException("Freset must be positive");

    ResetKineticsSettings::setFReset(static_cast<size_t>(fReset));
}

/**
 * @brief parse nreset_angular and set it in settings
 *
 * @details default value is 0
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throw InputFileException if nreset_angular is negative
 */
void ResetKineticsInputParser::parseNResetAngular(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto nResetAngular = stoi(lineElements[2]);

    if (nResetAngular < 0)
        throw InputFileException("Nreset_angular must be positive");

    ResetKineticsSettings::setNResetAngular(static_cast<size_t>(nResetAngular));
}

/**
 * @brief parse freset_angular and set it in settings
 *
 * @details default value is 0 but then set to UINT_MAX in setup
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throw InputFileException if freset_angular is negative
 */
void ResetKineticsInputParser::parseFResetAngular(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto fResetAngular = stoi(lineElements[2]);

    if (fResetAngular < 0)
        throw InputFileException("Freset_angular must be positive");

    ResetKineticsSettings::setFResetAngular(static_cast<size_t>(fResetAngular));
}

/**
 * @brief parse freset_force and set it in settings
 *
 * @details default value is 0 but then set to UINT_MAX in setup
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throw InputFileException if freset_force is negative
 */
void ResetKineticsInputParser::parseFResetForces(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto fResetForces = stoi(lineElements[2]);

    if (fResetForces < 0)
        throw InputFileException("Freset_force must be positive");

    ResetKineticsSettings::setFResetForces(static_cast<size_t>(fResetForces));
}