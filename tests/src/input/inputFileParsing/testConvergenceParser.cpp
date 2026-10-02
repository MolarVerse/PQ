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

#include <gtest/gtest.h>

#include "convergenceInputParser.hpp"
#include "convergenceSettings.hpp"
#include "exceptions.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

TEST_F(TestInputFileReader, parserEnergyConvergenceStrategy)
{
    EXPECT_EQ(
        settings::ConvSettings::getEnConvStrategy(),
        std::optional<ConvStrategy>()
    );

    auto       parser  = input::ConvInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("energy_conv_strategy"));
    const auto& parseFunc = funcMap.at("energy_conv_strategy");

    using enum ConvStrategy;

    auto lineElements =
        std::vector<std::string>{"energy-conv-strategy", "=", "loose"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ConvSettings::getEnConvStrategy(), LOOSE);

    _clearParser(parser);

    lineElements =
        std::vector<std::string>{"energy-conv-strategy", "=", "absolute"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ConvSettings::getEnConvStrategy(), ABSOLUTE);

    _clearParser(parser);

    lineElements =
        std::vector<std::string>{"energy-conv-strategy", "=", "relative"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ConvSettings::getEnConvStrategy(), RELATIVE);

    _clearParser(parser);

    lineElements =
        std::vector<std::string>{"energy-conv-strategy", "=", "rigorous"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ConvSettings::getEnConvStrategy(), RIGOROUS);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"energy-conv-strategy", "=", "notValid"}, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"energy-conv-strategy\" at line 0 "
        "in input file. Allowed values: rigorous, loose, absolute, relative"
    )
}

TEST_F(TestInputFileReader, parserUseEnergyConvergence)
{
    EXPECT_TRUE(settings::ConvSettings::getUseEnergyConv());

    auto       parser  = input::ConvInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("use_energy_conv"));
    const auto& parseFunc = funcMap.at("use_energy_conv");

    auto lineElements =
        std::vector<std::string>{"use-energy-conv", "=", "false"};
    parseFunc(lineElements, 0);
    EXPECT_FALSE(settings::ConvSettings::getUseEnergyConv());

    _clearParser(parser);

    lineElements = std::vector<std::string>{"use-energy-conv", "=", "true"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getUseEnergyConv());

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"use-energy-conv", "=", "notValid"}, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"use-energy-conv\" at line 0 in "
        "input file. Allowed values: on|off|true|false|yes|no"
    )
}

TEST_F(TestInputFileReader, parserUseForceConvergence)
{
    EXPECT_TRUE(settings::ConvSettings::getUseForceConv());

    auto       parser  = input::ConvInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("use_force_conv"));
    const auto& parseFunc = funcMap.at("use_force_conv");

    auto lineElements =
        std::vector<std::string>{"use-force-conv", "=", "false"};
    parseFunc(lineElements, 0);
    EXPECT_FALSE(settings::ConvSettings::getUseForceConv());

    _clearParser(parser);

    lineElements = std::vector<std::string>{"use-force-conv", "=", "true"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getUseForceConv());

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"use-force-conv", "=", "notValid"}, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"use-force-conv\" at line 0 in "
        "input file. Allowed values: on|off|true|false|yes|no"
    )
}

TEST_F(TestInputFileReader, parserUseMaxForceConvergence)
{
    EXPECT_TRUE(settings::ConvSettings::getUseMaxForceConv());

    auto       parser  = input::ConvInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("use_max_force_conv"));
    const auto& parseFunc = funcMap.at("use_max_force_conv");

    auto lineElements =
        std::vector<std::string>{"use-max-force-conv", "=", "false"};
    parseFunc(lineElements, 0);
    EXPECT_FALSE(settings::ConvSettings::getUseMaxForceConv());

    _clearParser(parser);

    lineElements = std::vector<std::string>{"use-max-force-conv", "=", "true"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getUseMaxForceConv());

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"use-max-force-conv", "=", "notValid"}, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"use-max-force-conv\" at line 0 "
        "in input file. Allowed values: on|off|true|false|yes|no"
    )
}

TEST_F(TestInputFileReader, parserUseRMSForceConvergence)
{
    EXPECT_TRUE(settings::ConvSettings::getUseRMSForceConv());

    auto       parser  = input::ConvInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("use_rms_force_conv"));
    const auto& parseFunc = funcMap.at("use_rms_force_conv");

    auto lineElements =
        std::vector<std::string>{"use-rms-force-conv", "=", "false"};
    parseFunc(lineElements, 0);
    EXPECT_FALSE(settings::ConvSettings::getUseRMSForceConv());

    _clearParser(parser);

    lineElements = std::vector<std::string>{"use-rms-force-conv", "=", "true"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getUseRMSForceConv());

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"use-rms-force-conv", "=", "notValid"}, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"use-rms-force-conv\" at line 0 "
        "in input file. Allowed values: on|off|true|false|yes|no"
    )
}

TEST_F(TestInputFileReader, parserEnergyConvergence)
{
    EXPECT_FALSE(settings::ConvSettings::getEnergyConv().has_value());

    auto       parser  = input::ConvInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("energy_conv"));
    const auto& parseFunc = funcMap.at("energy_conv");

    const auto lineElements =
        std::vector<std::string>{"energy-conv", "=", "1e-3"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getEnergyConv().has_value());
    EXPECT_EQ(settings::ConvSettings::getEnergyConv().value(), 1e-3);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"energy-conv", "=", "-1"}, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"energy-conv\" at line 0 in input file: "
        "failed validation with message Value must be greater than 0"
    )
}

TEST_F(TestInputFileReader, parserRelativeEnergyConvergence)
{
    EXPECT_FALSE(settings::ConvSettings::getRelEnergyConv().has_value());

    auto       parser  = input::ConvInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rel_energy_conv"));
    const auto& parseFunc = funcMap.at("rel_energy_conv");

    const auto lineElements =
        std::vector<std::string>{"rel-energy-conv", "=", "1e-3"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getRelEnergyConv().has_value());
    EXPECT_EQ(settings::ConvSettings::getRelEnergyConv().value(), 1e-3);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"rel-energy-conv", "=", "-1"}, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"rel-energy-conv\" at line 0 in input "
        "file: failed validation with message Value must be greater than 0"
    )
}

TEST_F(TestInputFileReader, parserAbsoluteEnergyConvergence)
{
    EXPECT_FALSE(settings::ConvSettings::getAbsEnergyConv().has_value());

    auto       parser  = input::ConvInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("abs_energy_conv"));
    const auto& parseFunc = funcMap.at("abs_energy_conv");

    const auto lineElements =
        std::vector<std::string>{"abs-energy-conv", "=", "1e-3"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getAbsEnergyConv().has_value());
    EXPECT_EQ(settings::ConvSettings::getAbsEnergyConv().value(), 1e-3);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"abs-energy-conv", "=", "-1"}, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"abs-energy-conv\" at line 0 in input "
        "file: failed validation with message Value must be greater than 0"
    )
}

TEST_F(TestInputFileReader, parserForceConvergence)
{
    EXPECT_FALSE(settings::ConvSettings::getForceConv().has_value());

    auto       parser  = input::ConvInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("force_conv"));
    const auto& parseFunc = funcMap.at("force_conv");

    const auto lineElements =
        std::vector<std::string>{"force-conv", "=", "1e-3"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getForceConv().has_value());
    EXPECT_EQ(settings::ConvSettings::getForceConv().value(), 1e-3);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"force-conv", "=", "-1"}, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"force-conv\" at line 0 in input file: "
        "failed validation with message Value must be greater than 0"
    )
}

TEST_F(TestInputFileReader, parserMaxForceConvergence)
{
    EXPECT_FALSE(settings::ConvSettings::getMaxForceConv().has_value());

    auto       parser  = input::ConvInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("max_force_conv"));
    const auto& parseFunc = funcMap.at("max_force_conv");

    const auto lineElements =
        std::vector<std::string>{"max-force-conv", "=", "1e-3"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getMaxForceConv().has_value());
    EXPECT_EQ(settings::ConvSettings::getMaxForceConv().value(), 1e-3);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"max-force-conv", "=", "-1"}, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"max-force-conv\" at line 0 in input "
        "file: failed validation with message Value must be greater than 0"
    )
}

TEST_F(TestInputFileReader, parserRMSForceConvergence)
{
    EXPECT_FALSE(settings::ConvSettings::getRMSForceConv().has_value());

    auto       parser  = input::ConvInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rms_force_conv"));
    const auto& parseFunc = funcMap.at("rms_force_conv");

    const auto lineElements =
        std::vector<std::string>{"rms-force-conv", "=", "1e-3"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::ConvSettings::getRMSForceConv().has_value());
    EXPECT_EQ(settings::ConvSettings::getRMSForceConv().value(), 1e-3);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"rms-force-conv", "=", "-1"}, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"rms-force-conv\" at line 0 in input "
        "file: failed validation with message Value must be greater than 0"
    )
}
