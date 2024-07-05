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

#include "QMMMInputParser.hpp"

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

#include "exceptions.hpp"        // for InputFileException, customException
#include "qmmmSettings.hpp"      // for QMMMSettings
#include "stringUtilities.hpp"   // for toLowerCopy

using namespace input;
using namespace engine;
using namespace customException;
using namespace settings;
using namespace utilities;

/**
 * @brief Construct a new QMMMInputParser:: QMMMInputParser object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) qm_prog <string> 2) qm_script
 * <string>
 *
 * @param engine
 */
QMMMInputParser::QMMMInputParser(Engine &engine) : InputFileParser(engine)
{
    addKeyword(
        std::string("qm_center"),
        bind_front(&QMMMInputParser::parseQMCenter, this),
        false
    );
    addKeyword(
        std::string("qm_only_list"),
        bind_front(&QMMMInputParser::parseQMOnlyList, this),
        false
    );
    addKeyword(
        std::string("mm_only_list"),
        bind_front(&QMMMInputParser::parseMMOnlyList, this),
        false
    );
    addKeyword(
        std::string("qm_charges"),
        bind_front(&QMMMInputParser::parseUseQMCharges, this),
        false
    );
    addKeyword(
        std::string("qm_core_radius"),
        bind_front(&QMMMInputParser::parseQMCoreRadius, this),
        false
    );
    addKeyword(
        std::string("qmmm_layer_radius"),
        bind_front(&QMMMInputParser::parseQMCoreRadius, this),
        false
    );
    addKeyword(
        std::string("qmmm_smoothing_radius"),
        bind_front(&QMMMInputParser::parseQMCoreRadius, this),
        false
    );
}

/**
 * @brief parse external QM Center which should be used
 *
 * @param lineElements
 * @param lineNumber
 */
void QMMMInputParser::parseQMCenter(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    QMMMSettings::setQMCenterString(lineElements[2]);
}

/**
 * @brief parse list of atoms which should be treated with QM only
 *
 * @param lineElements
 * @param lineNumber
 */
void QMMMInputParser::parseQMOnlyList(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    QMMMSettings::setMMOnlyListString(lineElements[2]);
}

/**
 * @brief parse list of atoms which should be treated with MM only
 *
 * @param lineElements
 * @param lineNumber
 */
void QMMMInputParser::parseMMOnlyList(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    QMMMSettings::setMMOnlyListString(lineElements[2]);
}

/**
 * @brief parse if QM charges should be used
 *
 * @param lineElements
 * @param lineNumber
 */
void QMMMInputParser::parseUseQMCharges(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    auto use_qm_charges = toLowerCopy(lineElements[2]);

    if ("qm" == use_qm_charges)
        QMMMSettings::setUseQMCharges(true);

    else if ("mm" == use_qm_charges)
        QMMMSettings::setUseQMCharges(false);

    else
        throw InputFileException(std::format(
            "Invalid qm_charges \"{}\" in input file\n"
            "Possible values are: qm, mm",
            lineElements[2]
        ));

    throw UserInputException("Not implemented");
}

/**
 * @brief parse QM core radius
 *
 * @param lineElements
 * @param lineNumber
 */
void QMMMInputParser::parseQMCoreRadius(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto qmCoreRadius = std::stod(lineElements[2]);

    if (qmCoreRadius < 0.0)
        throw InputFileException(std::format(
            "Invalid {} {} in input file - must be a positive number",
            lineElements[0],
            lineElements[2]
        ));

    QMMMSettings::setQMCoreRadius(qmCoreRadius);

    throw UserInputException("Not implemented");
}

/**
 * @brief parse QMMM layer radius
 *
 * @param lineElements
 * @param lineNumber
 */
void QMMMInputParser::parseQMMMLayerRadius(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto qmmmLayerRadius = std::stod(lineElements[2]);

    if (qmmmLayerRadius < 0.0)
        throw InputFileException(std::format(
            "Invalid {} {} in input file - must be a positive number",
            lineElements[0],
            lineElements[2]
        ));

    QMMMSettings::setQMMMLayerRadius(qmmmLayerRadius);

    throw UserInputException("Not implemented");
}

/**
 * @brief parse QMMM smoothing radius
 *
 * @param lineElements
 * @param lineNumber
 */
void QMMMInputParser::parseQMMMSmoothingRadius(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto qmmmSmoothingRadius = std::stod(lineElements[2]);

    if (qmmmSmoothingRadius < 0.0)
        throw InputFileException(std::format(
            "Invalid {} {} in input file - must be a positive number",
            lineElements[0],
            lineElements[2]
        ));

    QMMMSettings::setQMMMSmoothingRadius(qmmmSmoothingRadius);

    throw UserInputException("Not implemented");
}