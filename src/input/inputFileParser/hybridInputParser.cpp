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

#include "hybridInputParser.hpp"

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

#include "exceptions.hpp"        // for InputFileException, customException
#include "hybridSettings.hpp"    // for HybridSettings
#include "stringUtilities.hpp"   // for toLowerCopy

using namespace input;
using namespace engine;
using namespace customException;
using namespace settings;
using namespace utilities;

/**
 * @brief Construct a new HybridInputParser:: HybridInputParser object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) qm_prog <string> 2) qm_script
 * <string>
 *
 * @param engine
 */
HybridInputParser::HybridInputParser(Engine &engine) : InputFileParser(engine)
{
    addKeyword(
        std::string("core_center"),
        bind_front(&HybridInputParser::parseCoreCenter, this),
        false
    );
    addKeyword(
        std::string("core_only_list"),
        bind_front(&HybridInputParser::parseCoreOnlyList, this),
        false
    );
    addKeyword(
        std::string("non_core_only_list"),
        bind_front(&HybridInputParser::parseNonCoreOnlyList, this),
        false
    );
    addKeyword(
        std::string("qm_charges"),
        bind_front(&HybridInputParser::parseUseQMCharges, this),
        false
    );
    addKeyword(
        std::string("core_radius"),
        bind_front(&HybridInputParser::parseCoreRadius, this),
        false
    );
    addKeyword(
        std::string("layer_radius"),
        bind_front(&HybridInputParser::parseLayerRadius, this),
        false
    );
    addKeyword(
        std::string("smoothing_radius"),
        bind_front(&HybridInputParser::parseSmoothingRadius, this),
        false
    );
}

/**
 * @brief parse atom index selection which defines the core region
 *
 * @param lineElements
 * @param lineNumber
 */
void HybridInputParser::parseCoreCenter(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    HybridSettings::setCoreCenterString(lineElements[2]);
}

/**
 * @brief parse list of atoms which should be treated as core region only
 *
 * @param lineElements
 * @param lineNumber
 */
void HybridInputParser::parseCoreOnlyList(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    HybridSettings::setCoreOnlyListString(lineElements[2]);
}

/**
 * @brief parse list of atoms which should be treated as non-core region only
 *
 * @param lineElements
 * @param lineNumber
 */
void HybridInputParser::parseNonCoreOnlyList(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    HybridSettings::setNonCoreOnlyListString(lineElements[2]);
}

/**
 * @brief parse if QM charges should be used
 *
 * @param lineElements
 * @param lineNumber
 */
void HybridInputParser::parseUseQMCharges(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    auto use_qm_charges = toLowerCopy(lineElements[2]);

    if ("qm" == use_qm_charges)
        HybridSettings::setUseQMCharges(true);

    else if ("mm" == use_qm_charges)
        HybridSettings::setUseQMCharges(false);

    else
        throw InputFileException(std::format(
            "Invalid qm_charges \"{}\" in input file\n"
            "Possible values are: qm, mm",
            lineElements[2]
        ));

    throw UserInputException("Not implemented");
}

/**
 * @brief parse core radius
 *
 * @param lineElements
 * @param lineNumber
 */
void HybridInputParser::parseCoreRadius(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto coreRadius = std::stod(lineElements[2]);

    if (coreRadius < 0.0)
        throw InputFileException(std::format(
            "Invalid {} {} in input file - must be a positive number",
            lineElements[0],
            lineElements[2]
        ));

    HybridSettings::setCoreRadius(coreRadius);

    throw UserInputException("Not implemented");
}

/**
 * @brief parse layer radius
 *
 * @param lineElements
 * @param lineNumber
 */
void HybridInputParser::parseLayerRadius(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto layerRadius = std::stod(lineElements[2]);

    if (layerRadius < 0.0)
        throw InputFileException(std::format(
            "Invalid {} {} in input file - must be a positive number",
            lineElements[0],
            lineElements[2]
        ));

    HybridSettings::setLayerRadius(layerRadius);

    throw UserInputException("Not implemented");
}

/**
 * @brief parse smoothing radius
 *
 * @param lineElements
 * @param lineNumber
 */
void HybridInputParser::parseSmoothingRadius(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto smoothingRadius = std::stod(lineElements[2]);

    if (smoothingRadius < 0.0)
        throw InputFileException(std::format(
            "Invalid {} {} in input file - must be a positive number",
            lineElements[0],
            lineElements[2]
        ));

    HybridSettings::setSmoothingRadius(smoothingRadius);

    throw UserInputException("Not implemented");
}