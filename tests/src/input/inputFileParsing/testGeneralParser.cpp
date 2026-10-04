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

#include <format>
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
    input::GeneralInputParser parser;
    std::vector<std::string>  lineElements = {"jobtype", "=", "mm-md"};
    input::GeneralInputParser::parseJobTypeForEngine(lineElements, 0);
    EXPECT_EQ(settings::GeneralSettings::getJobtype(), JobType::MM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMMActivated(), true);

    lineElements = {"jobtype", "=", "qm-md"};
    input::GeneralInputParser::parseJobTypeForEngine(lineElements, 0);
    EXPECT_EQ(settings::GeneralSettings::getJobtype(), JobType::QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isQMActivated(), true);

    lineElements = {"jobtype", "=", "qm-rpmd"};
    input::GeneralInputParser::parseJobTypeForEngine(lineElements, 0);
    EXPECT_EQ(
        settings::GeneralSettings::getJobtype(),
        JobType::RING_POLYMER_QM_MD
    );
    EXPECT_EQ(settings::GeneralSettings::isQMActivated(), true);
    EXPECT_EQ(settings::GeneralSettings::isRingPolymerMDActivated(), true);

    lineElements = {"jobtype", "=", "mm-opt"};
    input::GeneralInputParser::parseJobTypeForEngine(lineElements, 0);
    EXPECT_EQ(settings::GeneralSettings::getJobtype(), JobType::MM_OPT);
    EXPECT_EQ(settings::GeneralSettings::isOptJobType(), true);
    EXPECT_EQ(settings::GeneralSettings::isMMActivated(), true);

    lineElements = {"jobtype", "=", "mm-hessian"};
    input::GeneralInputParser::parseJobTypeForEngine(lineElements, 0);
    EXPECT_EQ(settings::GeneralSettings::getJobtype(), JobType::MM_HESSIAN);
    EXPECT_EQ(settings::GeneralSettings::isMMActivated(), true);

    lineElements = {"jobtype", "=", "notValid"};
    EXPECT_THROW_MSG(
        input::GeneralInputParser::parseJobTypeForEngine(lineElements, 0),
        exc::InputFileException,
        "Invalid jobtype \"notValid\" in input file - possible values are:\n"
        "- mm-opt\n"
        "- mm-hessian\n"
        "- mm-md\n"
        "- qm-md\n"
        "- qm-rpmd\n"
        "- qmmm-md\n"
    );

    EXPECT_NO_THROW(parser.parseJobType(lineElements, 0));

    settings::GeneralSettings::setIsRingPolymerMDActivated(true);
    settings::GeneralSettings::setJobtype(JobType::NONE);
    EXPECT_FALSE(settings::GeneralSettings::isRingPolymerMDActivated());
    EXPECT_EQ(JobTypeMeta::toString(JobType::NONE), "NONE");
}

/**
 * @brief tests parsing the "dim" command
 *
 */
TEST_F(TestInputFileReader, parseDimensionality)
{
    input::GeneralInputParser parser;
    std::vector<std::string>  lineElements = {"dim", "=", "3"};
    input::GeneralInputParser::parseDimensionality(lineElements, 0);
    EXPECT_EQ(settings::GeneralSettings::getDimensionality(), 3);

    lineElements = {"dim", "=", "3D"};
    input::GeneralInputParser::parseDimensionality(lineElements, 0);
    EXPECT_EQ(settings::GeneralSettings::getDimensionality(), 3);

    lineElements = {"dim", "=", "2"};
    EXPECT_THROW_MSG(
        parser.parseDimensionality(lineElements, 0),
        exc::InputFileException,
        "Invalid dimensionality \"2\" in input file\n"
        "Possible values are: 3, 3d"
    );

    lineElements = {"dim", "=", "1"};
    EXPECT_THROW_MSG(
        parser.parseDimensionality(lineElements, 0),
        exc::InputFileException,
        "Invalid dimensionality \"1\" in input file\n"
        "Possible values are: 3, 3d"
    );

    lineElements = {"dim", "=", "0"};
    EXPECT_THROW_MSG(
        parser.parseDimensionality(lineElements, 0),
        exc::InputFileException,
        "Invalid dimensionality \"0\" in input file\n"
        "Possible values are: 3, 3d"
    );
}

/**
 * @brief tests parsing the "floatingPointType" command
 *
 */
TEST_F(TestInputFileReader, parseFloatingPointType)
{
    input::GeneralInputParser parser;
    const auto                funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("floating_point_type"));
    const auto& parseFunc = funcMap.at("floating_point_type");

    std::vector<std::string> lineElements = {
        "floating_point_type",
        "=",
        "float"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::GeneralSettings::getFloatingPointType(), FPType::FLOAT);

    _clearParser(parser);

    lineElements = {"floating_point_type", "=", "double"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::GeneralSettings::getFloatingPointType(),
        FPType::DOUBLE
    );

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
    input::GeneralInputParser parser;

    std::vector<std::string> lineElements = {"random_seed", "=", "0"};
    input::GeneralInputParser::parseRandomSeed(lineElements, 0);
    EXPECT_EQ(settings::GeneralSettings::isRandomSeedSet(), true);
    EXPECT_EQ(settings::GeneralSettings::getRandomSeed(), 0);
    settings::GeneralSettings::setIsRandomSeedSet(false);

    lineElements = {"random_seed", "=", "+73"};
    input::GeneralInputParser::parseRandomSeed(lineElements, 0);
    EXPECT_EQ(settings::GeneralSettings::isRandomSeedSet(), true);
    EXPECT_EQ(settings::GeneralSettings::getRandomSeed(), 73);
    settings::GeneralSettings::setIsRandomSeedSet(false);

    lineElements = {"random_seed", "=", std::to_string(UINT32_MAX)};
    input::GeneralInputParser::parseRandomSeed(lineElements, 0);
    EXPECT_EQ(settings::GeneralSettings::isRandomSeedSet(), true);
    EXPECT_EQ(settings::GeneralSettings::getRandomSeed(), UINT32_MAX);
    settings::GeneralSettings::setIsRandomSeedSet(false);

    lineElements = {
        "random_seed",
        "=",
        std::to_string(static_cast<std::int64_t>(UINT32_MAX) + 1)
    };
    EXPECT_THROW_MSG(
        parser.parseRandomSeed(lineElements, 0),
        exc::InputFileException,
        std::format(
            "Random seed value \"{}\" is out of range.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            static_cast<std::int64_t>(UINT32_MAX) + 1,
            UINT32_MAX
        )
    );
    EXPECT_EQ(settings::GeneralSettings::isRandomSeedSet(), false);

    lineElements = {"random_seed", "=", "-1"};
    EXPECT_THROW_MSG(
        parser.parseRandomSeed(lineElements, 0),
        exc::InputFileException,
        std::format(
            "Random seed value \"{}\" is out of range.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            -1,
            UINT32_MAX
        )
    );
    EXPECT_EQ(settings::GeneralSettings::isRandomSeedSet(), false);

    lineElements = {"random_seed", "=", "seed"};
    EXPECT_THROW_MSG(
        parser.parseRandomSeed(lineElements, 0),
        exc::InputFileException,
        std::format(
            "Random seed value \"{}\" is invalid.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            "seed",
            UINT32_MAX
        )
    );
    EXPECT_EQ(settings::GeneralSettings::isRandomSeedSet(), false);

    lineElements = {"random_seed", "=", "3.14159"};
    EXPECT_THROW_MSG(
        parser.parseRandomSeed(lineElements, 0),
        exc::InputFileException,
        std::format(
            "Random seed value \"{}\" is invalid.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            3.14159,
            UINT32_MAX
        )
    );
    EXPECT_EQ(settings::GeneralSettings::isRandomSeedSet(), false);

    lineElements = {"random_seed", "=", "1e3"};
    EXPECT_THROW_MSG(
        parser.parseRandomSeed(lineElements, 0),
        exc::InputFileException,
        std::format(
            "Random seed value \"{}\" is invalid.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            "1e3",
            UINT32_MAX
        )
    );
    EXPECT_EQ(settings::GeneralSettings::isRandomSeedSet(), false);

    lineElements = {"random_seed", "=", "+"};
    EXPECT_THROW_MSG(
        parser.parseRandomSeed(lineElements, 0),
        exc::InputFileException,
        std::format(
            "Random seed value \"{}\" is invalid.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            "+",
            UINT32_MAX
        )
    );
    EXPECT_EQ(settings::GeneralSettings::isRandomSeedSet(), false);
}
