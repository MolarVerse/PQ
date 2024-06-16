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

TEST_F(TestInputFileReader, parserConvergenceStrategy)
{
    EXPECT_EQ(ConvSettings::getConvergenceStrategy(), ConvStrategy::RIGOROUS);

    auto parser = ConvInputParser(*_engine);

    auto lineElements = strings{"conv-strategy", "=", "loose"};
    parser.parseConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(ConvSettings::getConvergenceStrategy(), ConvStrategy::LOOSE);

    lineElements = strings{"conv-strategy", "=", "absolute"};
    parser.parseConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(ConvSettings::getConvergenceStrategy(), ConvStrategy::ABSOLUTE);

    lineElements = strings{"conv-strategy", "=", "relative"};
    parser.parseConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(ConvSettings::getConvergenceStrategy(), ConvStrategy::RELATIVE);

    lineElements = strings{"conv-strategy", "=", "rigorous"};
    parser.parseConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(ConvSettings::getConvergenceStrategy(), ConvStrategy::RIGOROUS);

    ASSERT_THROW_MSG(
        parser.parseConvergenceStrategy({"conv-strategy", "=", "notValid"}, 0),
        customException::InputFileException,
        "Unknown convergence strategy \"notValid\" in input file at line 0.\n"
        "Possible options are: rigorous, loose, absolute, relative"
    )
}

TEST_F(TestInputFileReader, parserEnergyConvergenceStrategy)
{
    EXPECT_EQ(
        ConvSettings::getEnergyConvergenceStrategy(),
        ConvStrategy::RIGOROUS
    );

    auto parser = ConvInputParser(*_engine);

    auto lineElements = strings{"energy-conv-strategy", "=", "loose"};
    parser.parseEnergyConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(
        ConvSettings::getEnergyConvergenceStrategy(),
        ConvStrategy::LOOSE
    );

    lineElements = strings{"energy-conv-strategy", "=", "absolute"};
    parser.parseEnergyConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(
        ConvSettings::getEnergyConvergenceStrategy(),
        ConvStrategy::ABSOLUTE
    );

    lineElements = strings{"energy-conv-strategy", "=", "relative"};
    parser.parseEnergyConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(
        ConvSettings::getEnergyConvergenceStrategy(),
        ConvStrategy::RELATIVE
    );

    lineElements = strings{"energy-conv-strategy", "=", "rigorous"};
    parser.parseEnergyConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(
        ConvSettings::getEnergyConvergenceStrategy(),
        ConvStrategy::RIGOROUS
    );

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

TEST_F(TestInputFileReader, parserForceConvergenceStrategy)
{
    EXPECT_EQ(
        ConvSettings::getForceConvergenceStrategy(),
        ConvStrategy::RIGOROUS
    );

    auto parser = ConvInputParser(*_engine);

    auto lineElements = strings{"force-conv-strategy", "=", "loose"};
    parser.parseForceConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(ConvSettings::getForceConvergenceStrategy(), ConvStrategy::LOOSE);

    lineElements = strings{"force-conv-strategy", "=", "absolute"};
    parser.parseForceConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(
        ConvSettings::getForceConvergenceStrategy(),
        ConvStrategy::ABSOLUTE
    );

    lineElements = strings{"force-conv-strategy", "=", "relative"};
    parser.parseForceConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(
        ConvSettings::getForceConvergenceStrategy(),
        ConvStrategy::RELATIVE
    );

    lineElements = strings{"force-conv-strategy", "=", "rigorous"};
    parser.parseForceConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(
        ConvSettings::getForceConvergenceStrategy(),
        ConvStrategy::RIGOROUS
    );

    ASSERT_THROW_MSG(
        parser.parseForceConvergenceStrategy(
            {"force-conv-strategy", "=", "notValid"},
            0
        ),
        customException::InputFileException,
        "Unknown force convergence strategy \"notValid\" in input file at "
        "line 0.\n"
        "Possible options are: rigorous, loose, absolute, relative"
    )
}

TEST_F(TestInputFileReader, parserUseEnergyConvergence)
{
    EXPECT_TRUE(ConvSettings::getUseEnergyConvergence());

    auto parser = ConvInputParser(*_engine);

    auto lineElements = strings{"use-energy-conv", "=", "false"};
    parser.parseUseEnergyConvergence(lineElements, 0);
    EXPECT_FALSE(ConvSettings::getUseEnergyConvergence());

    lineElements = strings{"use-energy-conv", "=", "true"};
    parser.parseUseEnergyConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getUseEnergyConvergence());

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
    EXPECT_TRUE(ConvSettings::getUseForceConvergence());

    auto parser = ConvInputParser(*_engine);

    auto lineElements = strings{"use-force-conv", "=", "false"};
    parser.parseUseForceConvergence(lineElements, 0);
    EXPECT_FALSE(ConvSettings::getUseForceConvergence());

    lineElements = strings{"use-force-conv", "=", "true"};
    parser.parseUseForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getUseForceConvergence());

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
    EXPECT_TRUE(ConvSettings::getUseMaxForceConvergence());

    auto parser = ConvInputParser(*_engine);

    auto lineElements = strings{"use-max-force-conv", "=", "false"};
    parser.parseUseMaxForceConvergence(lineElements, 0);
    EXPECT_FALSE(ConvSettings::getUseMaxForceConvergence());

    lineElements = strings{"use-max-force-conv", "=", "true"};
    parser.parseUseMaxForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getUseMaxForceConvergence());

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
    EXPECT_TRUE(ConvSettings::getUseRMSForceConvergence());

    auto parser = ConvInputParser(*_engine);

    auto lineElements = strings{"use-rms-force-conv", "=", "false"};
    parser.parseUseRMSForceConvergence(lineElements, 0);
    EXPECT_FALSE(ConvSettings::getUseRMSForceConvergence());

    lineElements = strings{"use-rms-force-conv", "=", "true"};
    parser.parseUseRMSForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getUseRMSForceConvergence());

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
    EXPECT_FALSE(ConvSettings::getEnergyConvergence().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"energy-conv", "=", "1e-3"};
    parser.parseEnergyConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getEnergyConvergence().has_value());
    EXPECT_EQ(ConvSettings::getEnergyConvergence().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseEnergyConvergence({"energy-conv", "=", "-1"}, 0),
        customException::InputFileException,
        "Energy convergence must be greater than 0.0 in input file at "
        "line 0."
    )
}

TEST_F(TestInputFileReader, parserRelativeEnergyConvergence)
{
    EXPECT_FALSE(ConvSettings::getRelEnergyConvergence().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"rel-energy-conv", "=", "1e-3"};
    parser.parseRelativeEnergyConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getRelEnergyConvergence().has_value());
    EXPECT_EQ(ConvSettings::getRelEnergyConvergence().value(), 1e-3);

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
    EXPECT_FALSE(ConvSettings::getAbsEnergyConvergence().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"abs-energy-conv", "=", "1e-3"};
    parser.parseAbsoluteEnergyConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getAbsEnergyConvergence().has_value());
    EXPECT_EQ(ConvSettings::getAbsEnergyConvergence().value(), 1e-3);

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
    EXPECT_FALSE(ConvSettings::getForceConvergence().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"force-conv", "=", "1e-3"};
    parser.parseForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getForceConvergence().has_value());
    EXPECT_EQ(ConvSettings::getForceConvergence().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseForceConvergence({"force-conv", "=", "-1"}, 0),
        customException::InputFileException,
        "Force convergence must be greater than 0.0 in input file at line 0."
    )
}

TEST_F(TestInputFileReader, parserRelativeForceConvergence)
{
    EXPECT_FALSE(ConvSettings::getRelForceConvergence().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"rel-force-conv", "=", "1e-3"};
    parser.parseRelativeForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getRelForceConvergence().has_value());
    EXPECT_EQ(ConvSettings::getRelForceConvergence().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseRelativeForceConvergence({"rel-force-conv", "=", "-1"}, 0),
        customException::InputFileException,
        "Relative force convergence must be greater than 0.0 in input file "
        "at line 0."
    )
}

TEST_F(TestInputFileReader, parserAbsoluteForceConvergence)
{
    EXPECT_FALSE(ConvSettings::getAbsForceConvergence().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"abs-force-conv", "=", "1e-3"};
    parser.parseAbsoluteForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getAbsForceConvergence().has_value());
    EXPECT_EQ(ConvSettings::getAbsForceConvergence().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseAbsoluteForceConvergence({"abs-force-conv", "=", "-1"}, 0),
        customException::InputFileException,
        "Absolute force convergence must be greater than 0.0 in input file "
        "at line 0."
    )
}

TEST_F(TestInputFileReader, parserMaxForceConvergence)
{
    EXPECT_FALSE(ConvSettings::getMaxForceConvergence().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"max-force-conv", "=", "1e-3"};
    parser.parseMaxForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getMaxForceConvergence().has_value());
    EXPECT_EQ(ConvSettings::getMaxForceConvergence().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseMaxForceConvergence({"max-force-conv", "=", "-1"}, 0),
        customException::InputFileException,
        "Max force convergence must be greater than 0.0 in input file at "
        "line 0."
    )
}

TEST_F(TestInputFileReader, parserRelativeMaxForceConvergence)
{
    EXPECT_FALSE(ConvSettings::getRelMaxForceConvergence().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"rel-max-force-conv", "=", "1e-3"};
    parser.parseRelativeMaxForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getRelMaxForceConvergence().has_value());
    EXPECT_EQ(ConvSettings::getRelMaxForceConvergence().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseRelativeMaxForceConvergence(
            {"rel-max-force-conv", "=", "-1"},
            0
        ),
        customException::InputFileException,
        "Relative max force convergence must be greater than 0.0 in input "
        "file at line 0."
    )
}

TEST_F(TestInputFileReader, parserAbsoluteMaxForceConvergence)
{
    EXPECT_FALSE(ConvSettings::getAbsMaxForceConvergence().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"abs-max-force-conv", "=", "1e-3"};
    parser.parseAbsoluteMaxForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getAbsMaxForceConvergence().has_value());
    EXPECT_EQ(ConvSettings::getAbsMaxForceConvergence().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseAbsoluteMaxForceConvergence(
            {"abs-max-force-conv", "=", "-1"},
            0
        ),
        customException::InputFileException,
        "Absolute max force convergence must be greater than 0.0 in input "
        "file at line 0."
    )
}

TEST_F(TestInputFileReader, parserRMSForceConvergence)
{
    EXPECT_FALSE(ConvSettings::getRMSForceConvergence().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"rms-force-conv", "=", "1e-3"};
    parser.parseRMSForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getRMSForceConvergence().has_value());
    EXPECT_EQ(ConvSettings::getRMSForceConvergence().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseRMSForceConvergence({"rms-force-conv", "=", "-1"}, 0),
        customException::InputFileException,
        "RMS force convergence must be greater than 0.0 in input file at "
        "line 0."
    )
}

TEST_F(TestInputFileReader, parserRelativeRMSForceConvergence)
{
    EXPECT_FALSE(ConvSettings::getRelRMSForceConvergence().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"rel-rms-force-conv", "=", "1e-3"};
    parser.parseRelativeRMSForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getRelRMSForceConvergence().has_value());
    EXPECT_EQ(ConvSettings::getRelRMSForceConvergence().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseRelativeRMSForceConvergence(
            {"rel-rms-force-conv", "=", "-1"},
            0
        ),
        customException::InputFileException,
        "Relative RMS force convergence must be greater than 0.0 in input "
        "file at line 0."
    )
}

TEST_F(TestInputFileReader, parserAbsoluteRMSForceConvergence)
{
    EXPECT_FALSE(ConvSettings::getAbsRMSForceConvergence().has_value());

    auto parser = ConvInputParser(*_engine);

    const auto lineElements = strings{"abs-rms-force-conv", "=", "1e-3"};
    parser.parseAbsoluteRMSForceConvergence(lineElements, 0);
    EXPECT_TRUE(ConvSettings::getAbsRMSForceConvergence().has_value());
    EXPECT_EQ(ConvSettings::getAbsRMSForceConvergence().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseAbsoluteRMSForceConvergence(
            {"abs-rms-force-conv", "=", "-1"},
            0
        ),
        customException::InputFileException,
        "Absolute RMS force convergence must be greater than 0.0 in input "
        "file at line 0."
    )
}