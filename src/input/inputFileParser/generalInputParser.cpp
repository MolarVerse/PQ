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

#include "generalInputParser.hpp"

#include <algorithm>    // for ranges::remove
#include <cctype>       // for std::isdigit
#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front
#include <limits>       // for numeric_limits

#include "engine.hpp"       // for Engine
#include "exceptions.hpp"   // for InputFileException, customException
#include "mmmdEngine.hpp"   // for MMMDEngine
#include "optEngine.hpp"    // for MMOptEngine
#include "qmmdEngine.hpp"   // for QMMDEngine
#include "ringPolymerqmmdEngine.hpp"   // for RingPolymerQMMDEngine
#include "settings.hpp"                // for Settings
#include "stringUtilities.hpp"         // for toLowerCopy

using namespace input;
using namespace settings;
using namespace utilities;
using namespace customException;
using namespace engine;
using std::format;

/**
 * @brief Construct a new Input File Parser General:: Input File Parser General
 * object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) jobtype <string> (required)
 *
 * @param engine
 */
GeneralInputParser::GeneralInputParser(Engine &engine) : InputFileParser(engine)
{
    addKeyword(
        std::string("jobtype"),
        bind_front(&GeneralInputParser::parseJobType, this),
        true
    );

    addKeyword(
        std::string("dim"),
        bind_front(&GeneralInputParser::parseDimensionality, this),
        false
    );

    addKeyword(
        std::string("floating_point_type"),
        bind_front(&GeneralInputParser::parseFloatingPointType, this),
        false
    );

    addKeyword(
        std::string("random_seed"),
        bind_front(&GeneralInputParser::parseRandomSeed, this),
        false
    );
}

/**
 * @brief parse jobtype of simulation left empty just to not parse it again
 * after engine is generated
 */
void GeneralInputParser::parseJobType(
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
 * @throw InputFileException if jobtype is not recognised
 */
void GeneralInputParser::parseJobTypeForEngine(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber,
    std::unique_ptr<Engine>        &engine
)
{
    using enum JobType;
    checkCommand(lineElements, lineNumber);

    const auto jobtype = toLowerAndReplaceDashesCopy(lineElements[2]);

    if (jobtype == "mm_opt")
    {
        Settings::setJobtype(MM_OPT);
        engine.reset(new OptEngine());
    }
    else if (jobtype == "mm_md")
    {
        Settings::setJobtype(MM_MD);
        engine.reset(new MMMDEngine());
    }
    else if (jobtype == "qm_md")
    {
        Settings::setJobtype(QM_MD);
        engine.reset(new QMMDEngine());
    }
    else if (jobtype == "qm_rpmd")
    {
        Settings::setJobtype(RING_POLYMER_QM_MD);
        engine.reset(new RingPolymerQMMDEngine());
    }
    else
        throw InputFileException(format(
            "Invalid jobtype \"{}\" in input file - possible values are:\n"
            "- mm-opt\n"
            "- mm-md\n"
            "- qm-md\n"
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
 * @throw InputFileException if dimensionality is not
 * recognised
 */
void GeneralInputParser::parseDimensionality(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    auto dimensionalityString = toLowerCopy(lineElements[2]);

    std::erase(dimensionalityString, 'd');

    const auto dimensionality = std::stoi(dimensionalityString);

    if (dimensionality == 3)
        Settings::setDimensionality(size_t(dimensionality));

    else
        throw InputFileException(format(
            "Invalid dimensionality \"{}\" in input file\n"
            "Possible values are: 3, 3d",
            lineElements[2]
        ));
}

/**
 * @brief parse floating point type of simulation
 *
 * @details Possible options are:
 * 1) float
 * 2) double
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throw InputFileException if floating point type is not
 * recognised
 */
void GeneralInputParser::parseFloatingPointType(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    using enum FPType;
    checkCommand(lineElements, lineNumber);

    const auto floatingPointType = toLowerCopy(lineElements[2]);

    if (floatingPointType == "float")
        Settings::setFloatingPointType(FLOAT);

    else if (floatingPointType == "double")
        Settings::setFloatingPointType(DOUBLE);

    else
        throw InputFileException(format(
            "Invalid floating point type \"{}\" in input file\n"
            "Possible values are: float, double",
            lineElements[2]
        ));
}

/**
 * @brief parse random seed value for PRNG
 *
 * @details value not set as default
 *
 * @param lineElements
 *
 * @throws InputFileException if random seed value is invalid, negative, or
 * exceeds uint_fast32_t range
 */
void GeneralInputParser::parseRandomSeed(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    constexpr auto maxRandomSeed = static_cast<long long>(UINT32_MAX);

    auto throwRangeError = [&maxRandomSeed](const auto &value)
    {
        throw InputFileException(format(
            "Random seed value \"{}\" is out of range.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            value,
            maxRandomSeed
        ));
    };

    auto throwValidityError = [&maxRandomSeed](const auto &value)
    {
        throw InputFileException(format(
            "Random seed value \"{}\" is invalid.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            value,
            maxRandomSeed
        ));
    };

    std::uint_fast32_t randomSeed;

    try
    {
        randomSeed = utilities::stringToUintFast32t(lineElements[2]);
    }
    catch (const std::invalid_argument &)
    {
        throwValidityError(lineElements[2]);
    }
    catch (const std::out_of_range &)
    {
        throwRangeError(lineElements[2]);
    }

    Settings::setIsRandomSeedSet(true);
    Settings::setRandomSeed(randomSeed);
}