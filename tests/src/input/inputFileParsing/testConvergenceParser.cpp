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

#include "convergenceInputParser.hpp"   // for InputFileParserOptimizer
#include "convergenceSettings.hpp"      // for ConvSettings
#include "exceptions.hpp"            // for InputFileException, customException
#include "gtest/gtest.h"             // for Message, TestPartResult
#include "inputFileParser.hpp"       // for readInput
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for ASSERT_THROW_MSG

using namespace input;
using namespace settings;

TEST_F(TestInputFileReader, parserEnergyConvergenceStrategy)
{
    EXPECT_EQ(
        ConvSettings::getEnergyConvStrategy(),
        std::optional<ConvStrategy>()
    );

    auto parser = ConvInputParser(*_engine);

    auto lineElements = strings{"energy-conv-strategy", "=", "loose"};
    parser.parseEnergyConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(ConvSettings::getEnergyConvStrategy(), ConvStrategy::LOOSE);

    lineElements = strings{"energy-conv-strategy", "=", "absolute"};
    parser.parseEnergyConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(ConvSettings::getEnergyConvStrategy(), ConvStrategy::ABSOLUTE);

    lineElements = strings{"energy-conv-strategy", "=", "relative"};
    parser.parseEnergyConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(ConvSettings::getEnergyConvStrategy(), ConvStrategy::RELATIVE);

    lineElements = strings{"energy-conv-strategy", "=", "rigorous"};
    parser.parseEnergyConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(ConvSettings::getEnergyConvStrategy(), ConvStrategy::RIGOROUS);

    ASSERT_THROW_MSG(
        parser.parseEnergyConvergenceStrategy(
            {"energy-conv-strategy", "=", "notValid"},
            0
        ),
        customException::InputFileException,
        "Unknown energy convergence strategy \"notValid\" in input file at "
        "line 0.\n"
        "Possible options are: rigorous, loose, absolute, relative"
    )
}

TEST_F(TestInputFileReader, parserUseEnergyConvergence)
{
    EXPECT_TRUE(ConvSettings::getUseEnergyConv());

    auto parser = ConvInputParser(*_engine);

    auto lineElements = strings{"use-energy-conv", "=", "false"};
    parser.parseUseEnergyConvergence(lineElements, 0);
    EXPECT_FALSE(ConvSettings::getUseEnergyConv());

    lineElements = strings{"use-energy-conv", "=", "true"};
    parser.parseUseEnergyConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getUseEnergyConv());

    ASSERT_THROW_MSG(
        parser
            .parseUseEnergyConvergence({"use-energy-conv", "=", "notValid"}, 0),
        customException::InputFileException,
        "Unknown option \"notValid\" for use-energy-conv in input file "
        "at line 0.\n"
        "Possible options are: true, false"
    )
}

TEST_F(TestInputFileReader, parserUseForceConvergence)
{
    EXPECT_TRUE(ConvSettings::getUseForceConv());

    auto parser = ConvInputParser(*_engine);

    auto lineElements = strings{"use-force-conv", "=", "false"};
    parser.parseUseForceConvergence(lineElements, 0);
    EXPECT_FALSE(ConvSettings::getUseForceConv());

    lineElements = strings{"use-force-conv", "=", "true"};
    parser.parseUseForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getUseForceConv());

    ASSERT_THROW_MSG(
        parser.parseUseForceConvergence({"use-force-conv", "=", "notValid"}, 0),
        customException::InputFileException,
        "Unknown option \"notValid\" for use-force-conv in input file "
        "at line 0.\n"
        "Possible options are: true, false"
    )
}

TEST_F(TestInputFileReader, parserUseMaxForceConvergence)
{
    EXPECT_TRUE(ConvSettings::getUseMaxForceConv());

    auto parser = ConvInputParser(*_engine);

    auto lineElements = strings{"use-max-force-conv", "=", "false"};
    parser.parseUseMaxForceConvergence(lineElements, 0);
    EXPECT_FALSE(ConvSettings::getUseMaxForceConv());

    lineElements = strings{"use-max-force-conv", "=", "true"};
    parser.parseUseMaxForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getUseMaxForceConv());

    ASSERT_THROW_MSG(
        parser.parseUseMaxForceConvergence(
            {"use-max-force-conv", "=", "notValid"},
            0
        ),
        customException::InputFileException,
        "Unknown option \"notValid\" for use-max-force-conv in input "
        "file "
        "at line 0.\n"
        "Possible options are: true, false"
    )
}

TEST_F(TestInputFileReader, parserUseRMSForceConvergence)
{
    EXPECT_TRUE(ConvSettings::getUseRMSForceConv());

    auto parser = ConvInputParser(*_engine);

    auto lineElements = strings{"use-rms-force-conv", "=", "false"};
    parser.parseUseRMSForceConvergence(lineElements, 0);
    EXPECT_FALSE(ConvSettings::getUseRMSForceConv());

    lineElements = strings{"use-rms-force-conv", "=", "true"};
    parser.parseUseRMSForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getUseRMSForceConv());

    ASSERT_THROW_MSG(
        parser.parseUseRMSForceConvergence(
            {"use-rms-force-conv", "=", "notValid"},
            0
        ),
        customException::InputFileException,
        "Unknown option \"notValid\" for use-rms-force-conv in input "
        "file "
        "at line 0.\n"
        "Possible options are: true, false"
    )
}

TEST_F(TestInputFileReader, parserEnergyConvergence)
{
    EXPECT_FALSE(ConvSettings::getEnergyConv().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"energy-conv", "=", "1e-3"};
    parser.parseEnergyConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getEnergyConv().has_value());
    EXPECT_EQ(ConvSettings::getEnergyConv().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseEnergyConvergence({"energy-conv", "=", "-1"}, 0),
        customException::InputFileException,
        "Energy convergence must be greater than 0.0 in input file at "
        "line 0."
    )
}

TEST_F(TestInputFileReader, parserRelativeEnergyConvergence)
{
    EXPECT_FALSE(ConvSettings::getRelEnergyConv().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"rel-energy-conv", "=", "1e-3"};
    parser.parseRelativeEnergyConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getRelEnergyConv().has_value());
    EXPECT_EQ(ConvSettings::getRelEnergyConv().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser
            .parseRelativeEnergyConvergence({"rel-energy-conv", "=", "-1"}, 0),
        customException::InputFileException,
        "Relative energy convergence must be greater than 0.0 in input file "
        "at line 0."
    )
}

TEST_F(TestInputFileReader, parserAbsoluteEnergyConvergence)
{
    EXPECT_FALSE(ConvSettings::getAbsEnergyConv().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"abs-energy-conv", "=", "1e-3"};
    parser.parseAbsoluteEnergyConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getAbsEnergyConv().has_value());
    EXPECT_EQ(ConvSettings::getAbsEnergyConv().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser
            .parseAbsoluteEnergyConvergence({"abs-energy-conv", "=", "-1"}, 0),
        customException::InputFileException,
        "Absolute energy convergence must be greater than 0.0 in input file "
        "at line 0."
    )
}

TEST_F(TestInputFileReader, parserForceConvergence)
{
    EXPECT_FALSE(ConvSettings::getForceConv().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"force-conv", "=", "1e-3"};
    parser.parseForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getForceConv().has_value());
    EXPECT_EQ(ConvSettings::getForceConv().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseForceConvergence({"force-conv", "=", "-1"}, 0),
        customException::InputFileException,
        "Force convergence must be greater than 0.0 in input file at line 0."
    )
}

TEST_F(TestInputFileReader, parserMaxForceConvergence)
{
    EXPECT_FALSE(ConvSettings::getMaxForceConv().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"max-force-conv", "=", "1e-3"};
    parser.parseMaxForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getMaxForceConv().has_value());
    EXPECT_EQ(ConvSettings::getMaxForceConv().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseMaxForceConvergence({"max-force-conv", "=", "-1"}, 0),
        customException::InputFileException,
        "Max force convergence must be greater than 0.0 in input file at "
        "line 0."
    )
}

TEST_F(TestInputFileReader, parserRMSForceConvergence)
{
    EXPECT_FALSE(ConvSettings::getRMSForceConv().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"rms-force-conv", "=", "1e-3"};
    parser.parseRMSForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getRMSForceConv().has_value());
    EXPECT_EQ(ConvSettings::getRMSForceConv().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseRMSForceConvergence({"rms-force-conv", "=", "-1"}, 0),
        customException::InputFileException,
        "RMS force convergence must be greater than 0.0 in input file at "
        "line 0."
    )
}