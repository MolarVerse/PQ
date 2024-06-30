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
InputFileParserQMMM::InputFileParserQMMM(engine::Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        std::string("qm_center"),
        bind_front(&InputFileParserQMMM::parseQMCenter, this),
        false
    );
    addKeyword(
        std::string("qm_only_list"),
        bind_front(&InputFileParserQMMM::parseQMOnlyList, this),
        false
    );
    addKeyword(
        std::string("mm_only_list"),
        bind_front(&InputFileParserQMMM::parseMMOnlyList, this),
        false
    );
    addKeyword(
        std::string("qm_charges"),
        bind_front(&InputFileParserQMMM::parseUseQMCharges, this),
        false
    );
    addKeyword(
        std::string("qm_core_radius"),
        bind_front(&InputFileParserQMMM::parseQMCoreRadius, this),
        false
    );
    addKeyword(
        std::string("qmmm_layer_radius"),
        bind_front(&InputFileParserQMMM::parseQMCoreRadius, this),
        false
    );
    addKeyword(
        std::string("qmmm_smoothing_radius"),
        bind_front(&InputFileParserQMMM::parseQMCoreRadius, this),
        false
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

/**
 * @brief parse list of atoms which should be treated with QM only
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserQMMM::parseQMOnlyList(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::QMMMSettings::setMMOnlyListString(lineElements[2]);
}

/**
 * @brief parse list of atoms which should be treated with MM only
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserQMMM::parseMMOnlyList(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::QMMMSettings::setMMOnlyListString(lineElements[2]);
}

/**
 * @brief parse if QM charges should be used
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserQMMM::parseUseQMCharges(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    auto use_qm_charges = utilities::toLowerCopy(lineElements[2]);

    if ("qm" == use_qm_charges)
        settings::QMMMSettings::setUseQMCharges(true);

    else if ("mm" == use_qm_charges)
        settings::QMMMSettings::setUseQMCharges(false);

    else
        throw customException::InputFileException(std::format(
            "Invalid qm_charges \"{}\" in input file\n"
            "Possible values are: qm, mm",
            lineElements[2]
        ));

    throw customException::UserInputException("Not implemented");
}

/**
 * @brief parse QM core radius
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserQMMM::parseQMCoreRadius(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto qmCoreRadius = std::stod(lineElements[2]);

    if (qmCoreRadius < 0.0)
        throw customException::InputFileException(std::format(
            "Invalid {} {} in input file - must be a positive number",
            lineElements[0],
            lineElements[2]
        ));

    settings::QMMMSettings::setQMCoreRadius(qmCoreRadius);

    throw customException::UserInputException("Not implemented");
}

/**
 * @brief parse QMMM layer radius
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserQMMM::parseQMMMLayerRadius(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto qmmmLayerRadius = std::stod(lineElements[2]);

    if (qmmmLayerRadius < 0.0)
        throw customException::InputFileException(std::format(
            "Invalid {} {} in input file - must be a positive number",
            lineElements[0],
            lineElements[2]
        ));

    settings::QMMMSettings::setQMMMLayerRadius(qmmmLayerRadius);

    throw customException::UserInputException("Not implemented");
}

/**
 * @brief parse QMMM smoothing radius
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserQMMM::parseQMMMSmoothingRadius(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto qmmmSmoothingRadius = std::stod(lineElements[2]);

    if (qmmmSmoothingRadius < 0.0)
        throw customException::InputFileException(std::format(
            "Invalid {} {} in input file - must be a positive number",
            lineElements[0],
            lineElements[2]
        ));

    settings::QMMMSettings::setQMMMSmoothingRadius(qmmmSmoothingRadius);

    throw customException::UserInputException("Not implemented");
}