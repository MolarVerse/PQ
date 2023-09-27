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

#include "inputFileParserGeneral.hpp"

#include "engine.hpp"                  // for Engine
#include "exceptions.hpp"              // for InputFileException, customException
#include "mmmdEngine.hpp"              // for MMMDEngine
#include "qmmdEngine.hpp"              // for QMMDEngine
#include "ringPolymerqmmdEngine.hpp"   // for RingPolymerQMMDEngine
#include "settings.hpp"                // for Settings
#include "stringUtilities.hpp"         // for toLowerCopy

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

using namespace readInput;

/**
 * @brief Construct a new Input File Parser General:: Input File Parser General object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) jobtype <string> (required)
 *
 * @param engine
 */
InputFileParserGeneral::InputFileParserGeneral(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("jobtype"), bind_front(&InputFileParserGeneral::parseJobType, this), true);
}

/**
 * @brief parse jobtype of simulation left empty just to not parse it again after engine is generated
 */
void InputFileParserGeneral::parseJobType(const std::vector<std::string> &, const size_t) {}

/**
 * @brief parse jobtype of simulation and set it in settings and reset engine unique_ptr
 *
 * @details Possible options are:
 * 1) mm-md
 * 2) qm-md
 *
 * @param lineElements
 * @param lineNumber
 * @param engine
 *
 * @throw customException::InputFileException if jobtype is not recognised
 */
void InputFileParserGeneral::parseJobTypeForEngine(const std::vector<std::string>  &lineElements,
                                                   const size_t                     lineNumber,
                                                   std::unique_ptr<engine::Engine> &engine)
{
    checkCommand(lineElements, lineNumber);

    const auto jobtype = utilities::toLowerCopy(lineElements[2]);

    if (jobtype == "mm-md")
    {
        settings::Settings::setJobtype("MMMD");
        settings::Settings::activateMM();
        engine.reset(new engine::MMMDEngine());
    }
    else if (jobtype == "qm-md")
    {
        settings::Settings::setJobtype("QMMD");
        settings::Settings::activateQM();
        engine.reset(new engine::QMMDEngine());
    }
    else if (jobtype == "qm-rpmd")
    {
        settings::Settings::setJobtype("Ring_Polymer_QMMD");
        settings::Settings::activateQM();
        settings::Settings::activateRingPolymerMD();
        engine.reset(new engine::RingPolymerQMMDEngine());
    }
    else
        throw customException::InputFileException(
            format("Invalid jobtype \"{}\" in input file - possible values are: mm-md, qm-md, qm-rpmd", lineElements[2]));
}