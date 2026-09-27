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

#include <gtest/gtest.h>   // for TEST_F, EXPECT_EQ, RUN_ALL_TESTS

#include "exceptions.hpp"       // for exc::InputFileException, customException
#include "optInputParser.hpp"   // for InputFileParserOptimizer
#include "optimizerSettings.hpp"     // for settings::OptimizerSettings
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for ASSERT_THROW_MSG

/**
 * @brief test parsing the optimizer input key
 *
 * @details Possible keys are:
 * - optimizer = steepest-descent
 *
 */
TEST_F(TestInputFileReader, parserOptimizer)
{
    using enum settings::OptimizerType;

    EXPECT_EQ(settings::OptimizerSettings::getOptimizer(), STEEPEST_DESCENT);

    settings::OptimizerSettings::setOptimizer("none");

    auto parser = input::OptInputParser();
    input::OptInputParser::parseOptimizer(
        {"optimizer", "=", "steepest-descent"},
        0
    );
    EXPECT_EQ(settings::OptimizerSettings::getOptimizer(), STEEPEST_DESCENT);

    input::OptInputParser::parseOptimizer({"optimizer", "=", "adam"}, 0);
    EXPECT_EQ(settings::OptimizerSettings::getOptimizer(), ADAM);

    ASSERT_THROW_MSG(
        parser.parseOptimizer({"optimizer", "=", "notValid"}, 0),
        exc::InputFileException,
        "Unknown optimizer method \"notValid\" in input file at line 0.\n"
        "Possible options are: steepest-descent, adam"
    )
}

/**
 * @brief parse the optimizer learning rate strategy
 *
 * @details Possible keys are:
 * - learning-rate-strategy = constant-decay
 * - learning-rate-strategy = constant
 * - learning-rate-strategy = exponential-decay
 *
 */
TEST_F(TestInputFileReader, parserLearningRateStrategy)
{
    using enum settings::LREnum;

    EXPECT_EQ(
        settings::OptimizerSettings::getLearningRateStrategy(),
        EXPONENTIAL_DECAY
    );

    settings::OptimizerSettings::setLearningRateStrategy("none");

    auto parser = input::OptInputParser();
    input::OptInputParser::parseLearningRateStrategy(
        {"learning-rate-strategy", "=", "constant-decay"},
        0
    );
    EXPECT_EQ(
        settings::OptimizerSettings::getLearningRateStrategy(),
        CONSTANT_DECAY
    );

    input::OptInputParser::parseLearningRateStrategy(
        {"learning-rate-strategy", "=", "constant"},
        0
    );
    EXPECT_EQ(settings::OptimizerSettings::getLearningRateStrategy(), CONSTANT);

    input::OptInputParser::parseLearningRateStrategy(
        {"learning-rate-strategy", "=", "exponential-decay"},
        0
    );
    EXPECT_EQ(
        settings::OptimizerSettings::getLearningRateStrategy(),
        EXPONENTIAL_DECAY
    );

    input::OptInputParser::parseLearningRateStrategy(
        {"learning-rate-strategy", "=", "lineSearch-wolfe"},
        0
    );
    EXPECT_EQ(
        settings::OptimizerSettings::getLearningRateStrategy(),
        LINESEARCH_WOLFE
    );

    input::OptInputParser::parseLearningRateStrategy(
        {"learning-rate-strategy", "=", "linesearch"},
        0
    );
    EXPECT_EQ(
        settings::OptimizerSettings::getLearningRateStrategy(),
        LINESEARCH_WOLFE
    );

    ASSERT_THROW_MSG(
        parser.parseLearningRateStrategy(
            {"learning-rate-strategy", "=", "notValid"},
            0
        ),
        exc::InputFileException,
        "Unknown learning rate strategy \"notValid\" in input file at line 0.\n"
        "Possible options are: constant, constant-decay, exponential-decay, "
        "linesearch (linesearch-wolfe)"
    )
}

/**
 * @brief parse the optimizer initial Learning Rate
 *
 * @details The initial learning rate must be greater than 0.0
 *
 */
TEST_F(TestInputFileReader, parserInitialLearningRate)
{
    EXPECT_EQ(
        settings::OptimizerSettings::getInitialLearningRate(),
        defaults::INITIAL_LEARNING_RATE_DEFAULT
    );

    settings::OptimizerSettings::setInitialLearningRate(0.0);

    auto parser = input::OptInputParser();
    input::OptInputParser::parseInitialLearningRate(
        {"initial-learning-rate", "=", "0.99"},
        0
    );
    EXPECT_EQ(settings::OptimizerSettings::getInitialLearningRate(), 0.99);

    ASSERT_THROW_MSG(
        parser.parseInitialLearningRate(
            {"initial-learning-rate", "=", "-0.99"},
            0
        ),
        exc::InputFileException,
        "Initial learning rate must be greater than 0.0 in input file at line "
        "0."
    )
}

/**
 * @brief parse the optimizer learning rate decay
 *
 * @details The learning rate decay must be greater than 0.0
 *
 */
TEST_F(TestInputFileReader, parserLearningRateDecay)
{
    EXPECT_EQ(
        settings::OptimizerSettings::getLearningRateDecay(),
        std::nullopt
    );

    settings::OptimizerSettings::setLearningRateDecay(0.0);

    auto parser = input::OptInputParser();
    input::OptInputParser::parseLearningRateDecay(
        {"learning-rate-decay", "=", "0.99"},
        0
    );
    EXPECT_EQ(settings::OptimizerSettings::getLearningRateDecay(), 0.99);

    ASSERT_THROW_MSG(
        parser.parseLearningRateDecay({"learning-rate-decay", "=", "-0.99"}, 0),
        exc::InputFileException,
        "Learning rate decay must be greater than 0.0 in input file at line 0."
    )
}

/**
 * @brief parse the optimizer learning rate decay factor
 *
 * @details The learning rate decay factor must be greater than 0.0
 *
 */
TEST_F(TestInputFileReader, parserMaxLearningRate)
{
    EXPECT_EQ(settings::OptimizerSettings::getMaxLearningRate(), std::nullopt);

    settings::OptimizerSettings::setMaxLearningRate(0.0);

    auto parser = input::OptInputParser();
    input::OptInputParser::parseMaxLearningRate(
        {"max-learning-rate", "=", "0.99"},
        0
    );
    EXPECT_EQ(settings::OptimizerSettings::getMaxLearningRate(), 0.99);

    ASSERT_THROW_MSG(
        parser.parseMaxLearningRate({"max-learning-rate", "=", "-0.99"}, 0),
        exc::InputFileException,
        "Maximum learning rate must be greater than 0.0 in input file at line "
        "0."
    )
}

/**
 * @brief parse the optimizer learning rate decay frequency
 *
 * @details The learning rate decay frequency must be greater than 0.0
 *
 */
TEST_F(TestInputFileReader, parserLRUpdateFrequency)
{
    EXPECT_EQ(
        settings::OptimizerSettings::getLRUpdateFrequency(),
        defaults::LR_UPDATE_FREQUENCY_DEFAULT
    );

    settings::OptimizerSettings::setLRUpdateFrequency(0);

    auto parser = input::OptInputParser();
    input::OptInputParser::parseLearningRateUpdateFreq(
        {"lr-update-frequency", "=", "100"},
        0
    );
    EXPECT_EQ(settings::OptimizerSettings::getLRUpdateFrequency(), 100);

    ASSERT_THROW_MSG(
        parser.parseLearningRateUpdateFreq(
            {"lr-update-frequency", "=", "-100"},
            0
        ),
        exc::InputFileException,
        "Learning rate update frequency must be greater than 0 in input file "
        "at line 0."
    )
}

/**
 * @brief parse the minimum learning rate
 *
 * @details The minimum learning rate must be greater than 0.0
 *
 */
TEST_F(TestInputFileReader, parserMinLearningRate)
{
    EXPECT_EQ(
        settings::OptimizerSettings::getMinLearningRate(),
        defaults::MIN_LEARNING_RATE_DEFAULT
    );

    settings::OptimizerSettings::setMinLearningRate(0.0);

    auto parser = input::OptInputParser();
    input::OptInputParser::parseMinLearningRate(
        {"min-learning-rate", "=", "0.99"},
        0
    );
    EXPECT_EQ(settings::OptimizerSettings::getMinLearningRate(), 0.99);

    ASSERT_THROW_MSG(
        parser.parseMinLearningRate({"min-learning-rate", "=", "-0.99"}, 0),
        exc::InputFileException,
        "Minimum learning rate must be greater than 0.0 in input file at line "
        "0."
    )
}
