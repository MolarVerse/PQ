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

#include "inputFileParserQMMM.hpp"

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

#include "exceptions.hpp"        // for InputFileException, customException
#include "qmmmSettings.hpp"      // for QMMMSettings
#include "stringUtilities.hpp"   // for toLowerCopy

using namespace input;

/**
 * @brief Construct a new InputFileParserQMMM:: InputFileParserQMMM object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) qm_prog <string> 2) qm_script
 * <string>
 *
 * @param engine
 */
InputFileParserQMMM::InputFileParserQMMM(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(
        std::string("qm_center"), bind_front(&InputFileParserQMMM::parseQMCenter, this), false
    );
}

/**
 * @brief parse external QM Center which should be used
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserQMMM::parseQMCenter(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    settings::QMMMSettings::setQMCenterString(lineElements[2]);
}