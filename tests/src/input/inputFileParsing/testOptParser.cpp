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

#include <string>   // for string, allocator

#include "exceptions.hpp"            // for InputFileException, customException
#include "gtest/gtest.h"             // for Message, TestPartResult
#include "inputFileParser.hpp"       // for readInput
#include "optInputParser.hpp"        // for InputFileParserOptimizer
#include "optimizerSettings.hpp"     // for OptimizerSettings
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for ASSERT_THROW_MSG

using namespace input;
using namespace settings;

TEST_F(TestInputFileReader, parserOptimizer)
{
    EXPECT_EQ(OptimizerSettings::getOptimizer(), Optimizer::STEEPEST_DESCENT);

    OptimizerSettings::setOptimizer("none");

    auto parser = OptInputParser(*_engine);
    parser.parseOptimizer({"optimizer", "=", "steepest-descent"}, 0);
    EXPECT_EQ(
        settings::OptimizerSettings::getOptimizer(),
        settings::Optimizer::STEEPEST_DESCENT
    );

    ASSERT_THROW_MSG(
        parser.parseOptimizer({"optimizer", "=", "notValid"}, 0),
        customException::InputFileException,
        "Unknown optimizer method \"notValid\" in input file at line 0.\n"
        "Possible options are: steepest-descent"
    )
}

TEST_F(TestInputFileReader, parseNumberOfEpochs)
{
    EXPECT_EQ(OptimizerSettings::getNumberOfEpochs(), 100);

    auto parser = OptInputParser(*_engine);
    parser.parseNumberOfEpochs({"n-iterations", "=", "1000"}, 0);
    EXPECT_EQ(settings::OptimizerSettings::getNumberOfEpochs(), 1000);

    ASSERT_THROW_MSG(
        parser.parseNumberOfEpochs({"n-iterations", "=", "-1000"}, 0),
        customException::InputFileException,
        "Number of epochs must be greater than 0 in input file at line 0."
    )
}

TEST_F(TestInputFileReader, parserLearningRateStrategy)
{
    EXPECT_EQ(
        OptimizerSettings::getLearningRateStrategy(),
        LearningRateStrategy::NONE
    );

    OptimizerSettings::setLearningRateStrategy("none");

    auto parser = OptInputParser(*_engine);
    parser.parseLearningRateStrategy(
        {"learning-rate-strategy", "=", "decay"},
        0
    );
    EXPECT_EQ(
        settings::OptimizerSettings::getLearningRateStrategy(),
        settings::LearningRateStrategy::DECAY
    );

    parser.parseLearningRateStrategy(
        {"learning-rate-strategy", "=", "constant"},
        0
    );
    EXPECT_EQ(
        settings::OptimizerSettings::getLearningRateStrategy(),
        settings::LearningRateStrategy::CONSTANT
    );

    ASSERT_THROW_MSG(
        parser.parseLearningRateStrategy(
            {"learning-rate-strategy", "=", "notValid"},
            0
        ),
        customException::InputFileException,
        "Unknown learning rate strategy \"notValid\" in input file at line 0.\n"
        "Possible options are: constant, decay"
    )
}

TEST_F(TestInputFileReader, parserInitialLearningRate)
{
    EXPECT_EQ(
        OptimizerSettings::getInitialLearningRate(),
        defaults::_INITIAL_LEARNING_RATE_DEFAULT_
    );

    OptimizerSettings::setInitialLearningRate(0.0);

    auto parser = OptInputParser(*_engine);
    parser.parseInitialLearningRate({"initial-learning-rate", "=", "0.99"}, 0);
    EXPECT_EQ(settings::OptimizerSettings::getInitialLearningRate(), 0.99);

    ASSERT_THROW_MSG(
        parser.parseInitialLearningRate(
            {"initial-learning-rate", "=", "-0.99"},
            0
        ),
        customException::InputFileException,
        "Initial learning rate must be greater than 0.0 in input file at line "
        "0."
    )
}