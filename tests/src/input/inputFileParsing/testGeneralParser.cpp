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

#include <string>
#include <vector>

#include "exceptions.hpp"
#include "generalInputParser.hpp"
#include "generalSettings.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

/**
 * @brief tests parsing the "jobtype" command
 *
 * @details if the jobtype is not valid it throws inputFileException - possible
 * jobtypes are: mm-md
 *
 */
TEST_F(TestInputFileReader, JobType)
{
    std::vector<std::string> lineElements = {"jobtype", "=", "mm-md"};
    GeneralSettings          generalSettings;
    input::GeneralInputParser::parseJobTypeForEngine(
        generalSettings,
        lineElements,
        0
    );
    EXPECT_EQ(generalSettings.getJobtype(), JobType::MM_MD);
    EXPECT_EQ(isMMActivated(generalSettings.getJobtype()), true);

    lineElements = {"jobtype", "=", "qm-md"};
    input::GeneralInputParser::parseJobTypeForEngine(
        generalSettings,
        lineElements,
        0
    );
    EXPECT_EQ(generalSettings.getJobtype(), JobType::QM_MD);
    EXPECT_EQ(isQMActivated(generalSettings.getJobtype()), true);

    lineElements = {"jobtype", "=", "qm-rpmd"};
    input::GeneralInputParser::parseJobTypeForEngine(
        generalSettings,
        lineElements,
        0
    );
    EXPECT_EQ(generalSettings.getJobtype(), JobType::RING_POLYMER_QM_MD);
    EXPECT_EQ(isQMActivated(generalSettings.getJobtype()), true);
    EXPECT_EQ(isRingPolymerMDActivated(generalSettings.getJobtype()), true);

    lineElements = {"jobtype", "=", "mm-opt"};
    input::GeneralInputParser::parseJobTypeForEngine(
        generalSettings,
        lineElements,
        0
    );
    EXPECT_EQ(generalSettings.getJobtype(), JobType::MM_OPT);
    EXPECT_EQ(isOptJobType(generalSettings.getJobtype()), true);
    EXPECT_EQ(isMMActivated(generalSettings.getJobtype()), true);

    lineElements = {"jobtype", "=", "mm-hessian"};
    input::GeneralInputParser::parseJobTypeForEngine(
        generalSettings,
        lineElements,
        0
    );
    EXPECT_EQ(generalSettings.getJobtype(), JobType::MM_HESSIAN);
    EXPECT_EQ(isMMActivated(generalSettings.getJobtype()), true);

    lineElements = {"jobtype", "=", "notValid"};
    EXPECT_THROW_MSG(
        input::GeneralInputParser::parseJobTypeForEngine(
            generalSettings,
            lineElements,
            0
        ),
        exc::InputFileException,
        "Invalid jobtype \"notValid\" in input file - possible values are:\n"
        "- mm-opt\n"
        "- mm-hessian\n"
        "- mm-md\n"
        "- qm-md\n"
        "- qm-rpmd\n"
        "- qmmm-md\n"
    );

    input::GeneralInputParser parser{generalSettings};
    EXPECT_NO_THROW(parser.parseJobType(lineElements, 0));
}

/**
 * @brief tests parsing the "dim" command
 *
 */
TEST_F(TestInputFileReader, parseDimensionality)
{
    GeneralSettings           generalSettings;
    input::GeneralInputParser parser{generalSettings};
    const auto                funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("dim"));

    const auto&              parseFunc    = funcMap.at("dim");
    std::vector<std::string> lineElements = {"dim", "=", "3"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(GeneralSettings::getDimensionality(), 3);

    _clearParser(parser);

    lineElements = {"dim", "=", "3D"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(GeneralSettings::getDimensionality(), 3);

    lineElements = {"dim", "=", "2"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"2\" for key \"dim\" at line 0 in input file: "
        "Dimensionality must be 3 or 3d"
    );

    _clearParser(parser);

    lineElements = {"dim", "=", "1"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"1\" for key \"dim\" at line 0 in input file: "
        "Dimensionality must be 3 or 3d"
    );

    _clearParser(parser);

    lineElements = {"dim", "=", "0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"dim\" at line 0 in input file: "
        "Dimensionality must be 3 or 3d"
    );
}

/**
 * @brief tests parsing the "floatingPointType" command
 *
 */
TEST_F(TestInputFileReader, parseFloatingPointType)
{
    GeneralSettings           generalSettings;
    input::GeneralInputParser parser{generalSettings};
    const auto                funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("floating_point_type"));
    const auto& parseFunc = funcMap.at("floating_point_type");

    std::vector<std::string> lineElements = {
        "floating_point_type",
        "=",
        "float"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(generalSettings.getFloatingPointType(), FPType::FLOAT);

    _clearParser(parser);

    lineElements = {"floating_point_type", "=", "double"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(generalSettings.getFloatingPointType(), FPType::DOUBLE);

    _clearParser(parser);

    lineElements = {"floating_point_type", "=", "notValid"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"floating_point_type\" at line 0 "
        "in input file. Allowed values: float, double"
    );
}

/**
 * @brief tests parsing the "random_seed" command
 *
 */
TEST_F(TestInputFileReader, parseRandomSeed)
{
    GeneralSettings           generalSettings;
    input::GeneralInputParser parser{generalSettings};
    const auto                funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("random_seed"));
    const auto& parseFunc = funcMap.at("random_seed");

    std::vector<std::string> lineElements = {"random_seed", "=", "0"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(generalSettings.isRandomSeedSetByUser(), true);
    EXPECT_EQ(generalSettings.getRandomSeed(), 0);
    generalSettings = GeneralSettings{};

    _clearParser(parser);

    lineElements = {"random_seed", "=", "+73"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(generalSettings.isRandomSeedSetByUser(), true);
    EXPECT_EQ(generalSettings.getRandomSeed(), 73);
    generalSettings = GeneralSettings{};

    _clearParser(parser);

    lineElements = {"random_seed", "=", std::to_string(UINT32_MAX)};
    parseFunc(lineElements, 0);
    EXPECT_EQ(generalSettings.isRandomSeedSetByUser(), true);
    EXPECT_EQ(generalSettings.getRandomSeed(), UINT32_MAX);
    generalSettings = GeneralSettings{};

    _clearParser(parser);

    lineElements = {
        "random_seed",
        "=",
        std::to_string(static_cast<std::int64_t>(UINT32_MAX) + 1)
    };
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"4294967296\" for key \"random_seed\" at line 0 in "
        "input file: failed validation with message Value must be between 0 "
        "and 4294967295"
    );
    EXPECT_EQ(generalSettings.isRandomSeedSetByUser(), false);

    _clearParser(parser);

    lineElements = {"random_seed", "=", "-1"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"random_seed\" at line 0 in input file. "
        "Value must be a positive integer"
    );
    EXPECT_EQ(generalSettings.isRandomSeedSetByUser(), false);

    _clearParser(parser);

    lineElements = {"random_seed", "=", "seed"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"seed\" for key \"random_seed\" at line 0 in input "
        "file. "
        "Value must be a positive integer"
    );
    EXPECT_EQ(generalSettings.isRandomSeedSetByUser(), false);

    _clearParser(parser);

    lineElements = {"random_seed", "=", "3.14159"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"3.14159\" for key \"random_seed\" at line 0 in input "
        "file. "
        "Value must be a positive integer"
    );
    EXPECT_EQ(generalSettings.isRandomSeedSetByUser(), false);

    _clearParser(parser);

    lineElements = {"random_seed", "=", "1e3"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"1e3\" for key \"random_seed\" at line 0 in input "
        "file. "
        "Value must be a positive integer"
    );
    EXPECT_EQ(generalSettings.isRandomSeedSetByUser(), false);

    _clearParser(parser);

    lineElements = {"random_seed", "=", "+"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"+\" for key \"random_seed\" at line 0 in input file. "
        "Value must be a positive integer"
    );
    EXPECT_EQ(generalSettings.isRandomSeedSetByUser(), false);
}
