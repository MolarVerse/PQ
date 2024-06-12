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

#include "exceptions.hpp"        // for InputFileException, customException
#include "gtest/gtest.h"         // for Message, TestPartResult
#include "inputFileParser.hpp"   // for readInput
#include "inputFileParserOptimizer.hpp"   // for InputFileParserOptimizer
#include "optimizerSettings.hpp"          // for OptimizerSettings
#include "testInputFileReader.hpp"        // for TestInputFileReader
#include "throwWithMessage.hpp"           // for ASSERT_THROW_MSG

using namespace input;
using namespace settings;

TEST_F(TestInputFileReader, parserOptimizer)
{
    EXPECT_EQ(OptimizerSettings::getOptimizer(), Optimizer::GRADIENT_DESCENT);

    OptimizerSettings::setOptimizer("none");

    auto parser = InputFileParserOptimizer(*_engine);
    parser.parseOptimizer({"optimizer", "=", "gradient-descent"}, 0);
    EXPECT_EQ(
        settings::OptimizerSettings::getOptimizer(),
        settings::Optimizer::GRADIENT_DESCENT
    );

    ASSERT_THROW_MSG(
        parser.parseOptimizer({"optimizer", "=", "notValid"}, 0),
        customException::InputFileException,
        "Unknown optimizer method \"notValid\" in input file at line 0.\n"
        "Possible options are: gradient-descent"
    )
}