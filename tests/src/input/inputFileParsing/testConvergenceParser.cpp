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

#include "convergenceInputParser.hpp"   // for InputFileParserOptimizer
#include "convergenceSettings.hpp"      // for ConvSettings
#include "exceptions.hpp"   // for exc::InputFileException, customException
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for ASSERT_THROW_MSG

TEST_F(TestInputFileReader, parserEnergyConvergenceStrategy)
{
    EXPECT_EQ(
        settings::ConvSettings::getEnConvStrategy(),
        std::optional<settings::ConvStrategy>()
    );

    auto parser = input::ConvInputParser{};

    using enum settings::ConvStrategy;

    auto lineElements =
        std::vector<std::string>{"energy-conv-strategy", "=", "loose"};
    input::ConvInputParser::parseEnergyConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(settings::ConvSettings::getEnConvStrategy(), LOOSE);

    lineElements =
        std::vector<std::string>{"energy-conv-strategy", "=", "absolute"};
    input::ConvInputParser::parseEnergyConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(settings::ConvSettings::getEnConvStrategy(), ABSOLUTE);

    lineElements =
        std::vector<std::string>{"energy-conv-strategy", "=", "relative"};
    input::ConvInputParser::parseEnergyConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(settings::ConvSettings::getEnConvStrategy(), RELATIVE);

    lineElements =
        std::vector<std::string>{"energy-conv-strategy", "=", "rigorous"};
    input::ConvInputParser::parseEnergyConvergenceStrategy(lineElements, 0);
    EXPECT_EQ(settings::ConvSettings::getEnConvStrategy(), RIGOROUS);

    ASSERT_THROW_MSG(
        parser.parseEnergyConvergenceStrategy(
            {"energy-conv-strategy", "=", "notValid"},
            0
        ),
        exc::InputFileException,
        "Unknown energy convergence strategy \"notValid\" in input file at "
        "line 0.\n"
        "Possible options are: rigorous, loose, absolute, relative"
    )
}

TEST_F(TestInputFileReader, parserUseEnergyConvergence)
{
    EXPECT_TRUE(settings::ConvSettings::getUseEnergyConv());

    auto parser = input::ConvInputParser{};

    auto lineElements =
        std::vector<std::string>{"use-energy-conv", "=", "false"};
    input::ConvInputParser::parseUseEnergyConvergence(lineElements, 0);
    EXPECT_FALSE(settings::ConvSettings::getUseEnergyConv());

    lineElements = std::vector<std::string>{"use-energy-conv", "=", "true"};
    input::ConvInputParser::parseUseEnergyConvergence(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getUseEnergyConv());

    ASSERT_THROW_MSG(
        parser
            .parseUseEnergyConvergence({"use-energy-conv", "=", "notValid"}, 0),
        exc::InputFileException,
        "Unknown option \"notValid\" for use-energy-conv in input file "
        "at line 0.\n"
        "Possible options are: true, false"
    )
}

TEST_F(TestInputFileReader, parserUseForceConvergence)
{
    EXPECT_TRUE(settings::ConvSettings::getUseForceConv());

    auto parser = input::ConvInputParser{};

    auto lineElements =
        std::vector<std::string>{"use-force-conv", "=", "false"};
    input::ConvInputParser::parseUseForceConvergence(lineElements, 0);
    EXPECT_FALSE(settings::ConvSettings::getUseForceConv());

    lineElements = std::vector<std::string>{"use-force-conv", "=", "true"};
    input::ConvInputParser::parseUseForceConvergence(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getUseForceConv());

    ASSERT_THROW_MSG(
        parser.parseUseForceConvergence({"use-force-conv", "=", "notValid"}, 0),
        exc::InputFileException,
        "Unknown option \"notValid\" for use-force-conv in input file "
        "at line 0.\n"
        "Possible options are: true, false"
    )
}

TEST_F(TestInputFileReader, parserUseMaxForceConvergence)
{
    EXPECT_TRUE(settings::ConvSettings::getUseMaxForceConv());

    auto parser = input::ConvInputParser{};

    auto lineElements =
        std::vector<std::string>{"use-max-force-conv", "=", "false"};
    input::ConvInputParser::parseUseMaxForceConvergence(lineElements, 0);
    EXPECT_FALSE(settings::ConvSettings::getUseMaxForceConv());

    lineElements = std::vector<std::string>{"use-max-force-conv", "=", "true"};
    input::ConvInputParser::parseUseMaxForceConvergence(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getUseMaxForceConv());

    ASSERT_THROW_MSG(
        parser.parseUseMaxForceConvergence(
            {"use-max-force-conv", "=", "notValid"},
            0
        ),
        exc::InputFileException,
        "Unknown option \"notValid\" for use-max-force-conv in input "
        "file "
        "at line 0.\n"
        "Possible options are: true, false"
    )
}

TEST_F(TestInputFileReader, parserUseRMSForceConvergence)
{
    EXPECT_TRUE(settings::ConvSettings::getUseRMSForceConv());

    auto parser = input::ConvInputParser{};

    auto lineElements =
        std::vector<std::string>{"use-rms-force-conv", "=", "false"};
    input::ConvInputParser::parseUseRMSForceConvergence(lineElements, 0);
    EXPECT_FALSE(settings::ConvSettings::getUseRMSForceConv());

    lineElements = std::vector<std::string>{"use-rms-force-conv", "=", "true"};
    input::ConvInputParser::parseUseRMSForceConvergence(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getUseRMSForceConv());

    ASSERT_THROW_MSG(
        parser.parseUseRMSForceConvergence(
            {"use-rms-force-conv", "=", "notValid"},
            0
        ),
        exc::InputFileException,
        "Unknown option \"notValid\" for use-rms-force-conv in input "
        "file "
        "at line 0.\n"
        "Possible options are: true, false"
    )
}

TEST_F(TestInputFileReader, parserEnergyConvergence)
{
    EXPECT_FALSE(settings::ConvSettings::getEnergyConv().has_value());

    auto parser = input::ConvInputParser{};

    const auto lineElements =
        std::vector<std::string>{"energy-conv", "=", "1e-3"};
    input::ConvInputParser::parseEnergyConvergence(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getEnergyConv().has_value());
    EXPECT_EQ(settings::ConvSettings::getEnergyConv().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseEnergyConvergence({"energy-conv", "=", "-1"}, 0),
        exc::InputFileException,
        "Energy convergence must be greater than 0.0 in input file at "
        "line 0."
    )
}

TEST_F(TestInputFileReader, parserRelativeEnergyConvergence)
{
    EXPECT_FALSE(settings::ConvSettings::getRelEnergyConv().has_value());

    auto parser = input::ConvInputParser{};

    const auto lineElements =
        std::vector<std::string>{"rel-energy-conv", "=", "1e-3"};
    input::ConvInputParser::parseRelativeEnergyConvergence(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getRelEnergyConv().has_value());
    EXPECT_EQ(settings::ConvSettings::getRelEnergyConv().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser
            .parseRelativeEnergyConvergence({"rel-energy-conv", "=", "-1"}, 0),
        exc::InputFileException,
        "Relative energy convergence must be greater than 0.0 in input file "
        "at line 0."
    )
}

TEST_F(TestInputFileReader, parserAbsoluteEnergyConvergence)
{
    EXPECT_FALSE(settings::ConvSettings::getAbsEnergyConv().has_value());

    auto parser = input::ConvInputParser{};

    const auto lineElements =
        std::vector<std::string>{"abs-energy-conv", "=", "1e-3"};
    input::ConvInputParser::parseAbsoluteEnergyConvergence(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getAbsEnergyConv().has_value());
    EXPECT_EQ(settings::ConvSettings::getAbsEnergyConv().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser
            .parseAbsoluteEnergyConvergence({"abs-energy-conv", "=", "-1"}, 0),
        exc::InputFileException,
        "Absolute energy convergence must be greater than 0.0 in input file "
        "at line 0."
    )
}

TEST_F(TestInputFileReader, parserForceConvergence)
{
    EXPECT_FALSE(settings::ConvSettings::getForceConv().has_value());

    auto parser = input::ConvInputParser{};

    const auto lineElements =
        std::vector<std::string>{"force-conv", "=", "1e-3"};
    input::ConvInputParser::parseForceConvergence(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getForceConv().has_value());
    EXPECT_EQ(settings::ConvSettings::getForceConv().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseForceConvergence({"force-conv", "=", "-1"}, 0),
        exc::InputFileException,
        "Force convergence must be greater than 0.0 in input file at line 0."
    )
}

TEST_F(TestInputFileReader, parserMaxForceConvergence)
{
    EXPECT_FALSE(settings::ConvSettings::getMaxForceConv().has_value());

    auto parser = input::ConvInputParser{};

    const auto lineElements =
        std::vector<std::string>{"max-force-conv", "=", "1e-3"};
    input::ConvInputParser::parseMaxForceConvergence(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getMaxForceConv().has_value());
    EXPECT_EQ(settings::ConvSettings::getMaxForceConv().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseMaxForceConvergence({"max-force-conv", "=", "-1"}, 0),
        exc::InputFileException,
        "Max force convergence must be greater than 0.0 in input file at "
        "line 0."
    )
}

TEST_F(TestInputFileReader, parserRMSForceConvergence)
{
    EXPECT_FALSE(settings::ConvSettings::getRMSForceConv().has_value());

    auto parser = input::ConvInputParser{};

    const auto lineElements =
        std::vector<std::string>{"rms-force-conv", "=", "1e-3"};
    input::ConvInputParser::parseRMSForceConvergence(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getRMSForceConv().has_value());
    EXPECT_EQ(settings::ConvSettings::getRMSForceConv().value(), 1e-3);

    ASSERT_THROW_MSG(
        parser.parseRMSForceConvergence({"rms-force-conv", "=", "-1"}, 0),
        exc::InputFileException,
        "RMS force convergence must be greater than 0.0 in input file at "
        "line 0."
    )
}
