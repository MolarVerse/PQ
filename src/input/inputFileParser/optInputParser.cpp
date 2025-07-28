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

#include "exceptions.hpp"          // for InputFileException
#include "optimizerSettings.hpp"   // for OptimizerSettings
#include "stringUtilities.hpp"     // for toLowerCopy

using namespace input;
using namespace settings;
using namespace customException;
using namespace utilities;
using namespace engine;

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
OptInputParser::OptInputParser(Engine &engine) : InputFileParser(engine)
{
    addKeyword(
        "optimizer",
        bind_front(&OptInputParser::parseOptimizer, this),
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
        bind_front(&OptInputParser::parseLearningRateUpdateFreq, this),
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
 * @throws InputFileException if the optimizer method is
 * unknown
 */
void OptInputParser::parseOptimizer(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto method = toLowerAndReplaceDashesCopy(lineElements[2]);

    using enum OptimizerType;

    if ("steepest_descent" == method)
        OptimizerSettings::setOptimizer(STEEPEST_DESCENT);

    else if ("adam" == method)
        OptimizerSettings::setOptimizer(ADAM);

    else
        throw InputFileException(
            std::format(
                "Unknown optimizer method \"{}\" in input file "
                "at line {}.\nPossible options are: steepest-descent, "
                "adam",
                lineElements[2],
                lineNumber
            )
        );
}

/**
 * @brief Parses the learning rate strategy
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws InputFileException if the learning rate strategy is
 * unknown
 */
void OptInputParser::parseLearningRateStrategy(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    using enum LREnum;
    checkCommand(lineElements, lineNumber);

    const auto strategy = toLowerAndReplaceDashesCopy(lineElements[2]);

    if ("constant" == strategy)
        OptimizerSettings::setLearningRateStrategy(CONSTANT);

    else if ("constant_decay" == strategy)
        OptimizerSettings::setLearningRateStrategy(CONSTANT_DECAY);

    else if ("exponential_decay" == strategy)
        OptimizerSettings::setLearningRateStrategy(EXPONENTIAL_DECAY);

    else if ("linesearch_wolfe" == strategy)
        OptimizerSettings::setLearningRateStrategy(LINESEARCH_WOLFE);

    else if ("linesearch" == strategy)
        OptimizerSettings::setLearningRateStrategy(LINESEARCH_WOLFE);

    else
        throw InputFileException(
            std::format(
                "Unknown learning rate strategy \"{}\" in input file "
                "at line {}.\nPossible options are: constant, "
                "constant-decay, exponential-decay, linesearch "
                "(linesearch-wolfe)",
                lineElements[2],
                lineNumber
            )
        );
}

/**
 * @brief Parses the initial learning rate
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws InputFileException if the initial learning rate is
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
        throw InputFileException(
            std::format(
                "Initial learning rate must be greater than 0.0 in input file "
                "at line {}.",
                lineNumber
            )
        );

    OptimizerSettings::setInitialLearningRate(initialLearningRate);
}

/**
 * @brief Parses the learning rate update frequency
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws InputFileException if the learning rate update
 * frequency is less than or equal to 0
 */
void OptInputParser::parseLearningRateUpdateFreq(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    const auto frequency = std::stoi(lineElements[2]);

    if (frequency <= 0)
        throw InputFileException(
            std::format(
                "Learning rate update frequency must be greater than 0 in "
                "input "
                "file at line {}.",
                lineNumber
            )
        );

    OptimizerSettings::setLRUpdateFrequency(size_t(frequency));
}

/**
 * @brief Parses the minimum learning rate
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws InputFileException if the minimum learning rate is
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
        throw InputFileException(
            std::format(
                "Minimum learning rate must be greater than 0.0 in input file "
                "at line {}.",
                lineNumber
            )
        );

    OptimizerSettings::setMinLearningRate(minLearningRate);
}

/**
 * @brief Parses the maximum learning rate
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws InputFileException if the maximum learning rate is
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
        throw InputFileException(
            std::format(
                "Maximum learning rate must be greater than 0.0 in input file "
                "at line {}.",
                lineNumber
            )
        );

    OptimizerSettings::setMaxLearningRate(maxLearningRate);
}

/**
 * @brief Parses the learning rate decay
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 *
 * @throws InputFileException if the learning rate decay is
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
        throw InputFileException(
            std::format(
                "Learning rate decay must be greater than 0.0 in input file "
                "at line {}.",
                lineNumber
            )
        );

    OptimizerSettings::setLearningRateDecay(decay);
}