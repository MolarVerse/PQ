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

#include "enums/optimizer.hpp"
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
    using enum OptimizerType;

    EXPECT_EQ(settings::OptimizerSettings::getOptimizer(), STEEPEST_DESCENT);

    auto       parser  = input::OptInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("optimizer"));
    const auto& parseFunc = funcMap.at("optimizer");

    parseFunc({"optimizer", "=", "steepest-descent"}, 0);
    EXPECT_EQ(settings::OptimizerSettings::getOptimizer(), STEEPEST_DESCENT);

    clearParser(parser);

    parseFunc({"optimizer", "=", "adam"}, 0);
    EXPECT_EQ(settings::OptimizerSettings::getOptimizer(), ADAM);

    clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"optimizer", "=", "notValid"}, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"optimizer\" at line 0 in input "
        "file. Allowed values: steepest_descent, adam"
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
    using enum LearningRate;

    EXPECT_EQ(
        settings::OptimizerSettings::getLearningRateStrategy(),
        EXPONENTIAL_DECAY
    );

    auto       parser  = input::OptInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("learning_rate_strategy"));
    const auto& parseFunc = funcMap.at("learning_rate_strategy");

    parseFunc({"learning-rate-strategy", "=", "constant-decay"}, 0);
    EXPECT_EQ(
        settings::OptimizerSettings::getLearningRateStrategy(),
        CONSTANT_DECAY
    );

    clearParser(parser);

    parseFunc({"learning-rate-strategy", "=", "constant"}, 0);
    EXPECT_EQ(settings::OptimizerSettings::getLearningRateStrategy(), CONSTANT);

    clearParser(parser);

    parseFunc({"learning-rate-strategy", "=", "exponential-decay"}, 0);
    EXPECT_EQ(
        settings::OptimizerSettings::getLearningRateStrategy(),
        EXPONENTIAL_DECAY
    );

    clearParser(parser);

    parseFunc({"learning-rate-strategy", "=", "lineSearch-wolfe"}, 0);
    EXPECT_EQ(
        settings::OptimizerSettings::getLearningRateStrategy(),
        LINESEARCH_WOLFE
    );

    clearParser(parser);

    parseFunc({"learning-rate-strategy", "=", "linesearch"}, 0);
    EXPECT_EQ(
        settings::OptimizerSettings::getLearningRateStrategy(),
        LINESEARCH_WOLFE
    );

    clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"learning-rate-strategy", "=", "notValid"}, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"learning-rate-strategy\" at line "
        "0 in input file. Allowed values: constant, constant_decay, "
        "exponential_decay, linesearch_wolfe, linesearch"
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

    auto       parser  = input::OptInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("initial_learning_rate"));
    const auto& parseFunc = funcMap.at("initial_learning_rate");

    parseFunc({"initial-learning-rate", "=", "0.99"}, 0);
    EXPECT_EQ(settings::OptimizerSettings::getInitialLearningRate(), 0.99);

    clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"initial-learning-rate", "=", "-0.99"}, 0),
        exc::InputFileException,
        "Invalid value \"-0.99\" for key \"initial-learning-rate\" at line 0 "
        "in input file: failed validation with message Value must be greater "
        "than 0"
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

    auto       parser  = input::OptInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("learning_rate_decay"));
    const auto& parseFunc = funcMap.at("learning_rate_decay");

    parseFunc({"learning-rate-decay", "=", "0.99"}, 0);
    EXPECT_EQ(settings::OptimizerSettings::getLearningRateDecay(), 0.99);

    clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"learning-rate-decay", "=", "-0.99"}, 0),
        exc::InputFileException,
        "Invalid value \"-0.99\" for key \"learning-rate-decay\" at line 0 in "
        "input file: failed validation with message Value must be greater than "
        "0"
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

    auto       parser  = input::OptInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("max_learning_rate"));
    const auto& parseFunc = funcMap.at("max_learning_rate");

    parseFunc({"max-learning-rate", "=", "0.99"}, 0);
    EXPECT_EQ(settings::OptimizerSettings::getMaxLearningRate(), 0.99);

    clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"max-learning-rate", "=", "-0.99"}, 0),
        exc::InputFileException,
        "Invalid value \"-0.99\" for key \"max-learning-rate\" at line 0 in "
        "input file: failed validation with message Value must be greater than "
        "0"
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

    auto       parser  = input::OptInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("learning_rate_update_freq"));
    const auto& parseFunc = funcMap.at("learning_rate_update_freq");

    parseFunc({"lr-update-frequency", "=", "100"}, 0);
    EXPECT_EQ(settings::OptimizerSettings::getLRUpdateFrequency(), 100);

    clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"lr-update-frequency", "=", "-100"}, 0),
        exc::InputFileException,
        "Invalid value \"-100\" for key \"learning-rate-update-freq\" at line "
        "0 in input file. Value must be a positive integer"
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

    auto       parser  = input::OptInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("min_learning_rate"));
    const auto& parseFunc = funcMap.at("min_learning_rate");

    parseFunc({"min-learning-rate", "=", "0.99"}, 0);
    EXPECT_EQ(settings::OptimizerSettings::getMinLearningRate(), 0.99);

    clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"min-learning-rate", "=", "-0.99"}, 0),
        exc::InputFileException,
        "Invalid value \"-0.99\" for key \"min-learning-rate\" at line 0 in "
        "input file: failed validation with message Value must be greater than "
        "0"
    )
}
