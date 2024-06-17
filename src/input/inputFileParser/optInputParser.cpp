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

#include "optInputParser.hpp"

#include <format>   // for std::format
#include <string>   // for std::string

#include "exceptions.hpp"          // for customException::InputFileException
#include "optimizerSettings.hpp"   // for OptimizerSettings
#include "stringUtilities.hpp"     // for utilities::toLowerCopy

using namespace input;
using namespace settings;

/**
 * @brief Constructor
 *
 * @details following keywords are added:
 * - optimizer <string>
 * - n-iterations <int>
 * - learning-rate-strategy <string>
 * - initial-learning-rate <double>
 *
 * @param engine The engine
 */
OptInputParser::OptInputParser(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(
        "optimizer",
        bind_front(&OptInputParser::parseOptimizer, this),
        false
    );

    addKeyword(
        "n-iterations",
        bind_front(&OptInputParser::parseNumberOfEpochs, this),
        false
    );

    addKeyword(
        "learning-rate-strategy",
        bind_front(&OptInputParser::parseLearningRateStrategy, this),
        false
    );

    addKeyword(
        "initial-learning-rate",
        bind_front(&OptInputParser::parseInitialLearningRate, this),
        false
    );

    addKeyword(
        "learning-rate-decay",
        bind_front(&OptInputParser::parseLearningRateDecay, this),
        false
    );

    addKeyword(
        "learning-rate-update-freq",
        bind_front(&OptInputParser::parseLearningRateUpdateFrequency, this),
        false
    );

    addKeyword(
        "min-learning-rate",
        bind_front(&OptInputParser::parseMinLearningRate, this),
        false
    );

    addKeyword(
        "max-learning-rate",
        bind_front(&OptInputParser::parseMaxLearningRate, this),
        false
    );
}

/**
 * @brief Parses the optimizer
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the optimizer method is
 * unknown
 */
void OptInputParser::parseOptimizer(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto method = utilities::toLowerCopy(lineElements[2]);

    if ("steepest-descent" == method)
        OptimizerSettings::setOptimizer(Optimizer::STEEPEST_DESCENT);
    else
        throw customException::InputFileException(std::format(
            "Unknown optimizer method \"{}\" in input file "
            "at line {}.\n"
            "Possible options are: steepest-descent",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parses the number of epochs
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the number of epochs is less
 * than or equal to 0
 */
void OptInputParser::parseNumberOfEpochs(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto numberOfEpochs = std::stoi(lineElements[2]);

    if (numberOfEpochs <= 0)
        throw customException::InputFileException(std::format(
            "Number of epochs must be greater than 0 in input file "
            "at line {}.",
            lineNumber
        ));

    OptimizerSettings::setNumberOfEpochs(size_t(numberOfEpochs));
}

/**
 * @brief Parses the learning rate strategy
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the learning rate strategy is
 * unknown
 */
void OptInputParser::parseLearningRateStrategy(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto strategy = utilities::toLowerCopy(lineElements[2]);

    if ("constant" == strategy)
        OptimizerSettings::setLearningRateStrategy(
            LearningRateStrategy::CONSTANT
        );

    else if ("constant-decay" == strategy)
        OptimizerSettings::setLearningRateStrategy(
            LearningRateStrategy::CONSTANT_DECAY
        );

    else
        throw customException::InputFileException(std::format(
            "Unknown learning rate strategy \"{}\" in input file "
            "at line {}.\n"
            "Possible options are: constant, constant-decay",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parses the initial learning rate
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the initial learning rate is
 * less than or equal to 0.0
 */
void OptInputParser::parseInitialLearningRate(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto initialLearningRate = std::stod(lineElements[2]);

    if (initialLearningRate <= 0.0)
        throw customException::InputFileException(std::format(
            "Initial learning rate must be greater than 0.0 in input file "
            "at line {}.",
            lineNumber
        ));

    OptimizerSettings::setInitialLearningRate(initialLearningRate);
}

/**
 * @brief Parses the learning rate update frequency
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the learning rate update
 * frequency is less than or equal to 0
 */
void OptInputParser::parseLearningRateUpdateFrequency(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto frequency = std::stoi(lineElements[2]);

    if (frequency <= 0)
        throw customException::InputFileException(std::format(
            "Learning rate update frequency must be greater than 0 in input "
            "file at line {}.",
            lineNumber
        ));

    OptimizerSettings::setLRUpdateFrequency(size_t(frequency));
}

/**
 * @brief Parses the minimum learning rate
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the minimum learning rate is
 * less than or equal to 0.0
 */
void OptInputParser::parseMinLearningRate(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto minLearningRate = std::stod(lineElements[2]);

    if (minLearningRate <= 0.0)
        throw customException::InputFileException(std::format(
            "Minimum learning rate must be greater than 0.0 in input file "
            "at line {}.",
            lineNumber
        ));

    OptimizerSettings::setMinLearningRate(minLearningRate);
}

/**
 * @brief Parses the maximum learning rate
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the maximum learning rate is
 * less than or equal to 0.0
 */
void OptInputParser::parseMaxLearningRate(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto maxLearningRate = std::stod(lineElements[2]);

    if (maxLearningRate <= 0.0)
        throw customException::InputFileException(std::format(
            "Maximum learning rate must be greater than 0.0 in input file "
            "at line {}.",
            lineNumber
        ));

    OptimizerSettings::setMaxLearningRate(maxLearningRate);
}

/**
 * @brief Parses the learning rate decay
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws customException::InputFileException if the learning rate decay is
 * less than or equal to 0.0
 */
void OptInputParser::parseLearningRateDecay(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto decay = std::stod(lineElements[2]);

    if (decay <= 0.0)
        throw customException::InputFileException(std::format(
            "Learning rate decay must be greater than 0.0 in input file "
            "at line {}.",
            lineNumber
        ));

    OptimizerSettings::setLearningRateDecay(decay);
}