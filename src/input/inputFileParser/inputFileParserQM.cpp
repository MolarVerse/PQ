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

#include "inputFileParserQM.hpp"

#include "exceptions.hpp"        // for InputFileException, customException
#include "qmSettings.hpp"        // for Settings
#include "stringUtilities.hpp"   // for toLowerCopy

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

using namespace input;

/**
 * @brief Construct a new InputFileParserQM:: InputFileParserQM object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) qm_prog <string>
 * 2) qm_script <string>
 *
 * @param engine
 */
InputFileParserQM::InputFileParserQM(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("qm_prog"), bind_front(&InputFileParserQM::parseQMMethod, this), false);
    addKeyword(std::string("qm_script"), bind_front(&InputFileParserQM::parseQMScript, this), false);
    addKeyword(std::string("qm_script_full_path"), bind_front(&InputFileParserQM::parseQMScriptFullPath, this), false);
    addKeyword(std::string("qm_loop_time_limit"), bind_front(&InputFileParserQM::parseQMLoopTimeLimit, this), false);
}

/**
 * @brief parse external QM Program which should be used
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserQM::parseQMMethod(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto method = utilities::toLowerCopy(lineElements[2]);

    if ("dftbplus" == method)
        settings::QMSettings::setQMMethod("dftbplus");

    else if ("pyscf" == method)
        settings::QMSettings::setQMMethod("pyscf");

    else if ("turbomole" == method)
        settings::QMSettings::setQMMethod("turbomole");

    else
        throw customException::InputFileException(std::format(
            "Invalid qm_prog \"{}\" in input file - possible values are: dftbplus, pyscf, turbomole", lineElements[2]));
}

/**
 * @brief parse external QM Script name
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserQM::parseQMScript(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    settings::QMSettings::setQMScript(lineElements[2]);
}

/**
 * @brief parse external QM script name
 *
 * @details this keyword is used for singularity builds to ensure that the user knows
 * what he is doing. With a singularity build the script has to be accessed from outside of
 * the container and therefore the general keyword qm_script is not applicable.
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserQM::parseQMScriptFullPath(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    settings::QMSettings::setQMScriptFullPath(lineElements[2]);
}

/**
 * @brief parse the time limit for the QM loop
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserQM::parseQMLoopTimeLimit(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    settings::QMSettings::setQMLoopTimeLimit(std::stod(lineElements[2]));
}