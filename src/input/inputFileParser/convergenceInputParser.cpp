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

#include "convergenceInputParser.hpp"

#include <format>   // for std::format
#include <string>   // for std::string

#include "convergenceSettings.hpp"   // for ConvSettings
#include "exceptions.hpp"            // for customException::InputFileException
#include "stringUtilities.hpp"       // for utilities::toLowerCopy

using namespace input;
using namespace settings;

/**
 * @brief Constructor
 *
 * @details following keywords are added:
 * - conv-strategy <string>
 * - energy-conv-strategy <string>
 * - force-conv-strategy <string>
 *
 * - use-energy-conv <bool>
 * - use-force-conv <bool>
 * - use-max-force-conv <bool>
 * - use-rms-force-conv <bool>
 *
 * - energy-conv <double>
 * - rel-energy-conv <double>
 * - abs-energy-conv <double>
 *
 * - force-conv <double>
 * - rel-force-conv <double>
 * - abs-force-conv <double>
 *
 * - max-force-conv <double>
 * - rel-max-force-conv <double>
 * - abs-max-force-conv <double>
 *
 * - rms-force-conv <double>
 * - rel-rms-force-conv <double>
 * - abs-rms-force-conv <double>
 *
 * @param engine The engine
 */
ConvInputParser::ConvInputParser(engine::Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        "conv-strategy",
        bind_front(&ConvInputParser::parseConvergenceStrategy, this),
        false
    );

    addKeyword(
        "energy-conv-strategy",
        bind_front(&ConvInputParser::parseEnergyConvergenceStrategy, this),
        false
    );

    addKeyword(
        "force-conv-strategy",
        bind_front(&ConvInputParser::parseForceConvergenceStrategy, this),
        false
    );

    addKeyword(
        "use-energy-conv",
        bind_front(&ConvInputParser::parseUseEnergyConvergence, this),
        false
    );

    addKeyword(
        "use-force-conv",
        bind_front(&ConvInputParser::parseUseForceConvergence, this),
        false
    );

    addKeyword(
        "use-max-force-conv",
        bind_front(&ConvInputParser::parseUseMaxForceConvergence, this),
        false
    );

    addKeyword(
        "use-rms-force-conv",
        bind_front(&ConvInputParser::parseUseRMSForceConvergence, this),
        false
    );

    addKeyword(
        "energy-conv",
        bind_front(&ConvInputParser::parseEnergyConvergence, this),
        false
    );

    addKeyword(
        "rel-energy-conv",
        bind_front(&ConvInputParser::parseRelativeEnergyConvergence, this),
        false
    );

    addKeyword(
        "abs-energy-conv",
        bind_front(&ConvInputParser::parseAbsoluteEnergyConvergence, this),
        false
    );

    addKeyword(
        "force-conv",
        bind_front(&ConvInputParser::parseForceConvergence, this),
        false
    );

    addKeyword(
        "rel-force-conv",
        bind_front(&ConvInputParser::parseRelativeForceConvergence, this),
        false
    );

    addKeyword(
        "abs-force-conv",
        bind_front(&ConvInputParser::parseAbsoluteForceConvergence, this),
        false
    );

    addKeyword(
        "max-force-conv",
        bind_front(&ConvInputParser::parseMaxForceConvergence, this),
        false
    );

    addKeyword(
        "rel-max-force-conv",
        bind_front(&ConvInputParser::parseRelativeMaxForceConvergence, this),
        false
    );

    addKeyword(
        "abs-max-force-conv",
        bind_front(&ConvInputParser::parseAbsoluteMaxForceConvergence, this),
        false
    );

    addKeyword(
        "rms-force-conv",
        bind_front(&ConvInputParser::parseRMSForceConvergence, this),
        false
    );

    addKeyword(
        "rel-rms-force-conv",
        bind_front(&ConvInputParser::parseRelativeRMSForceConvergence, this),
        false
    );

    addKeyword(
        "abs-rms-force-conv",
        bind_front(&ConvInputParser::parseAbsoluteRMSForceConvergence, this),
        false
    );
}

/**
 * @brief parses the convergence strategy
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 */
void ConvInputParser::parseConvergenceStrategy(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto strategy = utilities::toLowerCopy(lineElements[2]);

    if ("rigorous" == strategy)
        ConvSettings::setConvStrategy(ConvStrategy::RIGOROUS);

    else if ("loose" == strategy)
        ConvSettings::setConvStrategy(ConvStrategy::LOOSE);

    else if ("absolute" == strategy)
        ConvSettings::setConvStrategy(ConvStrategy::ABSOLUTE);

    else if ("relative" == strategy)
        ConvSettings::setConvStrategy(ConvStrategy::RELATIVE);

    else
        throw customException::InputFileException(std::format(
            "Unknown convergence strategy \"{}\" in input file "
            "at line {}.\n"
            "Possible options are: rigorous, loose, absolute, relative",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parses the energy convergence strategy
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 */
void ConvInputParser::parseEnergyConvergenceStrategy(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto strategy = utilities::toLowerCopy(lineElements[2]);

    if ("rigorous" == strategy)
        ConvSettings::setEnergyConvStrategy(ConvStrategy::RIGOROUS);

    else if ("loose" == strategy)
        ConvSettings::setEnergyConvStrategy(ConvStrategy::LOOSE);

    else if ("absolute" == strategy)
        ConvSettings::setEnergyConvStrategy(ConvStrategy::ABSOLUTE);

    else if ("relative" == strategy)
        ConvSettings::setEnergyConvStrategy(ConvStrategy::RELATIVE);

    else
        throw customException::InputFileException(std::format(
            "Unknown energy convergence strategy \"{}\" in input file "
            "at line {}.\n"
            "Possible options are: rigorous, loose, absolute, relative",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parses the force convergence strategy
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 */
void ConvInputParser::parseForceConvergenceStrategy(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto strategy = utilities::toLowerCopy(lineElements[2]);

    if ("rigorous" == strategy)
        ConvSettings::setForceConvStrategy(ConvStrategy::RIGOROUS);

    else if ("loose" == strategy)
        ConvSettings::setForceConvStrategy(ConvStrategy::LOOSE);

    else if ("absolute" == strategy)
        ConvSettings::setForceConvStrategy(ConvStrategy::ABSOLUTE);

    else if ("relative" == strategy)
        ConvSettings::setForceConvStrategy(ConvStrategy::RELATIVE);

    else
        throw customException::InputFileException(std::format(
            "Unknown force convergence strategy \"{}\" in input file "
            "at line {}.\n"
            "Possible options are: rigorous, loose, absolute, relative",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parses the use energy convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 */
void ConvInputParser::parseUseEnergyConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto useEnergyConvergence = utilities::toLowerCopy(lineElements[2]);

    if ("true" == useEnergyConvergence)
        ConvSettings::setUseEnergyConv(true);

    else if ("false" == useEnergyConvergence)
        ConvSettings::setUseEnergyConv(false);

    else
        throw customException::InputFileException(std::format(
            "Unknown option \"{}\" for use-energy-conv in input file "
            "at line {}.\n"
            "Possible options are: true, false",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parses the use force convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 */
void ConvInputParser::parseUseForceConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto useForceConvergence = utilities::toLowerCopy(lineElements[2]);

    if ("true" == useForceConvergence)
        ConvSettings::setUseForceConv(true);

    else if ("false" == useForceConvergence)
        ConvSettings::setUseForceConv(false);

    else
        throw customException::InputFileException(std::format(
            "Unknown option \"{}\" for use-force-conv in input file "
            "at line {}.\n"
            "Possible options are: true, false",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parses the use max force convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 */
void ConvInputParser::parseUseMaxForceConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto useMaxForceConvergence = utilities::toLowerCopy(lineElements[2]);

    if ("true" == useMaxForceConvergence)
        ConvSettings::setUseMaxForceConv(true);

    else if ("false" == useMaxForceConvergence)
        ConvSettings::setUseMaxForceConv(false);

    else
        throw customException::InputFileException(std::format(
            "Unknown option \"{}\" for use-max-force-conv in input file "
            "at line {}.\n"
            "Possible options are: true, false",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parses the use RMS force convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 */
void ConvInputParser::parseUseRMSForceConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto useRMSForceConvergence = utilities::toLowerCopy(lineElements[2]);

    if ("true" == useRMSForceConvergence)
        ConvSettings::setUseRMSForceConv(true);

    else if ("false" == useRMSForceConvergence)
        ConvSettings::setUseRMSForceConv(false);

    else
        throw customException::InputFileException(std::format(
            "Unknown option \"{}\" for use-rms-force-conv in input file "
            "at line {}.\n"
            "Possible options are: true, false",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parses the energy convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the energy convergence is
 * less than or equal to 0.0
 */
void ConvInputParser::parseEnergyConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto energyConvergence = std::stod(lineElements[2]);

    if (energyConvergence <= 0.0)
        throw customException::InputFileException(std::format(
            "Energy convergence must be greater than 0.0 in input file "
            "at line {}.",
            lineNumber
        ));

    ConvSettings::setEnergyConv(energyConvergence);
}

/**
 * @brief Parses the relative energy convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the relative energy
 * convergence is less than or equal to 0.0
 */
void ConvInputParser::parseRelativeEnergyConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto relativeEnergyConvergence = std::stod(lineElements[2]);

    if (relativeEnergyConvergence <= 0.0)
        throw customException::InputFileException(std::format(
            "Relative energy convergence must be greater than 0.0 in input "
            "file "
            "at line {}.",
            lineNumber
        ));

    ConvSettings::setRelEnergyConv(relativeEnergyConvergence);
}

/**
 * @brief Parses the absolute energy convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the absolute energy
 * convergence is less than or equal to 0.0
 */
void ConvInputParser::parseAbsoluteEnergyConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto absoluteEnergyConvergence = std::stod(lineElements[2]);

    if (absoluteEnergyConvergence <= 0.0)
        throw customException::InputFileException(std::format(
            "Absolute energy convergence must be greater than 0.0 in input "
            "file "
            "at line {}.",
            lineNumber
        ));

    ConvSettings::setAbsEnergyConv(absoluteEnergyConvergence);
}

/**
 * @brief Parses the force convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the force convergence is
 * less than or equal to 0.0
 */
void ConvInputParser::parseForceConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto forceConvergence = std::stod(lineElements[2]);

    if (forceConvergence <= 0.0)
        throw customException::InputFileException(std::format(
            "Force convergence must be greater than 0.0 in input file "
            "at line {}.",
            lineNumber
        ));

    ConvSettings::setForceConv(forceConvergence);
}

/**
 * @brief Parses the relative force convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the relative force
 * convergence is less than or equal to 0.0
 */
void ConvInputParser::parseRelativeForceConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto relativeForceConvergence = std::stod(lineElements[2]);

    if (relativeForceConvergence <= 0.0)
        throw customException::InputFileException(std::format(
            "Relative force convergence must be greater than 0.0 in input file "
            "at line {}.",
            lineNumber
        ));

    ConvSettings::setRelForceConv(relativeForceConvergence);
}

/**
 * @brief Parses the absolute force convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the absolute force
 * convergence is less than or equal to 0.0
 */
void ConvInputParser::parseAbsoluteForceConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto absoluteForceConvergence = std::stod(lineElements[2]);

    if (absoluteForceConvergence <= 0.0)
        throw customException::InputFileException(std::format(
            "Absolute force convergence must be greater than 0.0 in input file "
            "at line {}.",
            lineNumber
        ));

    ConvSettings::setAbsForceConv(absoluteForceConvergence);
}

/**
 * @brief Parses the max force convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the max force convergence is
 * less than or equal to 0.0
 */
void ConvInputParser::parseMaxForceConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto maxForceConvergence = std::stod(lineElements[2]);

    if (maxForceConvergence <= 0.0)
        throw customException::InputFileException(std::format(
            "Max force convergence must be greater than 0.0 in input file "
            "at line {}.",
            lineNumber
        ));

    ConvSettings::setMaxForceConv(maxForceConvergence);
}

/**
 * @brief Parses the relative max force convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the relative max force
 * convergence is less than or equal to 0.0
 */
void ConvInputParser::parseRelativeMaxForceConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto relativeMaxForceConvergence = std::stod(lineElements[2]);

    if (relativeMaxForceConvergence <= 0.0)
        throw customException::InputFileException(std::format(
            "Relative max force convergence must be greater than 0.0 in input "
            "file "
            "at line {}.",
            lineNumber
        ));

    ConvSettings::setRelMaxForceConv(relativeMaxForceConvergence);
}

/**
 * @brief Parses the absolute max force convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the absolute max force
 * convergence is less than or equal to 0.0
 */
void ConvInputParser::parseAbsoluteMaxForceConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto absoluteMaxForceConvergence = std::stod(lineElements[2]);

    if (absoluteMaxForceConvergence <= 0.0)
        throw customException::InputFileException(std::format(
            "Absolute max force convergence must be greater than 0.0 in input "
            "file "
            "at line {}.",
            lineNumber
        ));

    ConvSettings::setAbsMaxForceConv(absoluteMaxForceConvergence);
}

/**
 * @brief Parses the RMS force convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the RMS force convergence is
 * less than or equal to 0.0
 */
void ConvInputParser::parseRMSForceConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto rmsForceConvergence = std::stod(lineElements[2]);

    if (rmsForceConvergence <= 0.0)
        throw customException::InputFileException(std::format(
            "RMS force convergence must be greater than 0.0 in input file "
            "at line {}.",
            lineNumber
        ));

    ConvSettings::setRMSForceConv(rmsForceConvergence);
}

/**
 * @brief Parses the relative RMS force convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the relative RMS force
 * convergence is less than or equal to 0.0
 */
void ConvInputParser::parseRelativeRMSForceConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto relativeRMSForceConvergence = std::stod(lineElements[2]);

    if (relativeRMSForceConvergence <= 0.0)
        throw customException::InputFileException(std::format(
            "Relative RMS force convergence must be greater than 0.0 in input "
            "file "
            "at line {}.",
            lineNumber
        ));

    ConvSettings::setRelRMSForceConv(relativeRMSForceConvergence);
}

/**
 * @brief Parses the absolute RMS force convergence
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the absolute RMS force
 * convergence is less than or equal to 0.0
 */
void ConvInputParser::parseAbsoluteRMSForceConvergence(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto absoluteRMSForceConvergence = std::stod(lineElements[2]);

    if (absoluteRMSForceConvergence <= 0.0)
        throw customException::InputFileException(std::format(
            "Absolute RMS force convergence must be greater than 0.0 in input "
            "file "
            "at line {}.",
            lineNumber
        ));

    ConvSettings::setAbsRMSForceConv(absoluteRMSForceConvergence);
}