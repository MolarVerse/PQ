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

#include "inputFileParserGeneral.hpp"

#include <algorithm>    // for ranges::remove
#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

#include "engine.hpp"         // for Engine
#include "exceptions.hpp"     // for InputFileException, customException
#include "mmmdEngine.hpp"     // for MMMDEngine
#include "optEngine.hpp"      // for MMOptEngine
#include "qmmdEngine.hpp"     // for QMMDEngine
#include "qmmmmdEngine.hpp"   // for QMMMMDEngine
#include "ringPolymerqmmdEngine.hpp"   // for RingPolymerQMMDEngine
#include "settings.hpp"                // for Settings
#include "stringUtilities.hpp"         // for toLowerCopy

using namespace input;

/**
 * @brief Construct a new Input File Parser General:: Input File Parser General
 * object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) jobtype <string> (required)
 *
 * @param engine
 */
InputFileParserGeneral::InputFileParserGeneral(engine::Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        std::string("jobtype"),
        bind_front(&InputFileParserGeneral::parseJobType, this),
        true
    );
    addKeyword(
        std::string("dim"),
        bind_front(&InputFileParserGeneral::parseDimensionality, this),
        false
    );
}

/**
 * @brief parse jobtype of simulation left empty just to not parse it again
 * after engine is generated
 */
void InputFileParserGeneral::parseJobType(
    const std::vector<std::string> &,
    const size_t
)
{
}

/**
 * @brief parse jobtype of simulation and set it in settings and reset engine
 * unique_ptr
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
void InputFileParserGeneral::parseJobTypeForEngine(
    const std::vector<std::string>  &lineElements,
    const size_t                     lineNumber,
    std::unique_ptr<engine::Engine> &engine
)
{
    checkCommand(lineElements, lineNumber);

    const auto jobtype = utilities::toLowerCopy(lineElements[2]);

    if (jobtype == "mm-opt")
    {
        settings::Settings::setJobtype("MMOPT");
        engine.reset(new engine::OptEngine());
    }
    else if (jobtype == "mm-md")
    {
        settings::Settings::setJobtype("MMMD");
        engine.reset(new engine::MMMDEngine());
    }
    else if (jobtype == "qm-md")
    {
        settings::Settings::setJobtype("QMMD");
        engine.reset(new engine::QMMDEngine());
    }
    else if (jobtype == "qmmm-md")
    {
        settings::Settings::setJobtype("QMMMMD");
        engine.reset(new engine::QMMMMDEngine());
    }
    else if (jobtype == "qm-rpmd")
    {
        settings::Settings::setJobtype("Ring_Polymer_QMMD");
        engine.reset(new engine::RingPolymerQMMDEngine());
    }
    else
        throw customException::InputFileException(format(
            "Invalid jobtype \"{}\" in input file - possible values are:\n"
            "- mm-opt\n"
            "- mm-md\n"
            "- qm-md\n"
            "- qmmm-md\n"
            "- qm-rpmd\n",
            lineElements[2]
        ));
}

/**
 * @brief parse dimensionality of simulation
 *
 * @details Possible options are:
 * 1) 3
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throw customException::InputFileException if dimensionality is not
 * recognised
 */
void InputFileParserGeneral::parseDimensionality(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    auto dimensionalityString = utilities::toLowerCopy(lineElements[2]);

    std::erase(dimensionalityString, 'd');

    const auto dimensionality = std::stoi(dimensionalityString);

    if (dimensionality == 3)
        settings::Settings::setDimensionality(size_t(dimensionality));
    else
        throw customException::InputFileException(format(
            "Invalid dimensionality \"{}\" in input file\n"
            "Possible values are: 3, 3d",
            lineElements[2]
        ));
}