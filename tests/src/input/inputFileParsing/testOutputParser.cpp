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

#include <gtest/gtest.h>   // for TestInfo (ptr only), EXPECT_EQ

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "exceptions.hpp"           // for InputFileException
#include "outputFileSettings.hpp"   // for OutputFileSettings
#include "outputInputParser.hpp"
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for EXPECT_THROW_MSG

/**
 * @brief tests parsing the "outputfreq" command
 *
 * @details if the outputfreq is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseOutputFreq)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("output_freq"));
    const auto &parseFunc = funcMap.at("output_freq");

    std::vector<std::string> lineElements = {"output_freq", "=", "1000"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getOutputFrequency(), 1000);

    _clearParser(parser);

    lineElements = {"output_freq", "=", "-1000"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-1000\" for key \"output_freq\" at line 0 in input "
        "file. Value must be a positive integer"
    );
}

/**
 * @brief tests parsing the "file_prefix" command
 *
 */
TEST_F(TestInputFileReader, testParseFilePrefix)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("file_prefix"));
    const auto &parseFunc = funcMap.at("file_prefix");

    const std::vector<std::string> lineElements = {
        "file_prefix",
        "=",
        "prefix"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getFilePrefix(), "prefix");
}

/**
 * @brief tests parsing the "output_file" command
 *
 */
TEST_F(TestInputFileReader, testParseLogFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("output_file"));
    const auto &parseFunc = funcMap.at("output_file");

    _fileName                             = "log.txt";
    std::vector<std::string> lineElements = {"output_file", "=", _fileName};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getLogFileName(), _fileName);
}

/**
 * @brief tests parsing the "info_file" command
 *
 */
TEST_F(TestInputFileReader, testParseInfoFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("info_file"));
    const auto &parseFunc = funcMap.at("info_file");

    _fileName                             = "info.txt";
    std::vector<std::string> lineElements = {"info_file", "=", "info.txt"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getInfoFileName(), "info.txt");
}

/**
 * @brief tests parsing the "energy_file" command
 *
 */
TEST_F(TestInputFileReader, testParseEnergyFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("energy_file"));
    const auto &parseFunc = funcMap.at("energy_file");

    _fileName                             = "energy.txt";
    std::vector<std::string> lineElements = {"energy_file", "=", _fileName};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getEnergyFileName(), _fileName);
}

/**
 * @brief tests parsing the "instant_energy_file" command
 *
 */
TEST_F(TestInputFileReader, testParseInstantEnergyFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("instant_energy_file"));
    const auto &parseFunc = funcMap.at("instant_energy_file");

    _fileName                                   = "instant_energy.txt";
    const std::vector<std::string> lineElements = {
        "instant_energy_file",
        "=",
        _fileName
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::OutputFileSettings::getInstantEnergyFileName(),
        _fileName
    );
}

/**
 * @brief tests parsing the "traj_file" command
 *
 */
TEST_F(TestInputFileReader, testParseTrajectoryFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("traj_file"));
    const auto &parseFunc = funcMap.at("traj_file");

    _fileName                             = "trajectory.xyz";
    std::vector<std::string> lineElements = {"traj_file", "=", _fileName};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getTrajectoryFileName(), _fileName);
}

/**
 * @brief tests parsing the "hybrid_center_file" command
 *
 */
TEST_F(TestInputFileReader, testParseHybridCenterFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("hybrid_center_file"));
    const auto &parseFunc = funcMap.at("hybrid_center_file");

    _fileName                                   = "center.xyz";
    const std::vector<std::string> lineElements = {
        "hybrid_center_file",
        "=",
        _fileName
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::OutputFileSettings::getHybridCenterFileName(),
        _fileName
    );
}

/**
 * @brief tests parsing the "velocity_file" command
 *
 */
TEST_F(TestInputFileReader, testVelocityFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("vel_file"));
    const auto &parseFunc = funcMap.at("vel_file");

    _fileName                             = "velocity.xyz";
    std::vector<std::string> lineElements = {"vel_file", "=", _fileName};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getVelocityFileName(), _fileName);
}

/**
 * @brief tests parsing the "force_file" command
 *
 */
TEST_F(TestInputFileReader, testForceFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("force_file"));
    const auto &parseFunc = funcMap.at("force_file");

    _fileName                             = "force.xyz";
    std::vector<std::string> lineElements = {"force_file", "=", _fileName};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getForceFileName(), _fileName);
}

/**
 * @brief tests parsing the "restart_file" command
 *
 */
TEST_F(TestInputFileReader, testParseRestartFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("restart_file"));
    const auto &parseFunc = funcMap.at("restart_file");

    _fileName                             = "restart.xyz";
    std::vector<std::string> lineElements = {"restart_file", "=", _fileName};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getRestartFileName(), _fileName);
}

/**
 * @brief tests parsing the "charge_file" command
 *
 */
TEST_F(TestInputFileReader, testChargeFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("charge_file"));
    const auto &parseFunc = funcMap.at("charge_file");

    _fileName                             = "charge.xyz";
    std::vector<std::string> lineElements = {"charge_file", "=", _fileName};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getChargeFileName(), _fileName);
}

/**
 * @brief tests parsing the "momentum_file" command
 *
 */
TEST_F(TestInputFileReader, testMomentumFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("momentum_file"));
    const auto &parseFunc = funcMap.at("momentum_file");

    _fileName                                   = "momentum.xyz";
    const std::vector<std::string> lineElements = {
        "momentum_file",
        "=",
        _fileName
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getMomentumFileName(), _fileName);
}

/**
 * @brief tests parsing the "virial_file" command
 *
 */
TEST_F(TestInputFileReader, testVirialFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("virial_file"));
    const auto &parseFunc = funcMap.at("virial_file");

    _fileName                                   = "viri.xyz";
    const std::vector<std::string> lineElements = {
        "virial_file",
        "=",
        _fileName
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getVirialFileName(), _fileName);
}

/**
 * @brief tests parsing the "stress_file" command
 *
 */
TEST_F(TestInputFileReader, testStressFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("stress_file"));
    const auto &parseFunc = funcMap.at("stress_file");

    _fileName                                   = "stress.xyz";
    const std::vector<std::string> lineElements = {
        "stress_file",
        "=",
        _fileName
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getStressFileName(), _fileName);
}

/**
 * @brief tests parsing the "box_file" command
 *
 */
TEST_F(TestInputFileReader, testBoxFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("box_file"));
    const auto &parseFunc = funcMap.at("box_file");

    _fileName                                   = "box.xyz";
    const std::vector<std::string> lineElements = {"box_file", "=", _fileName};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getBoxFileName(), _fileName);
}

/**
 * @brief tests parsing the "timings_file" command
 *
 */
TEST_F(TestInputFileReader, testTimingsFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("timings_file"));
    const auto &parseFunc = funcMap.at("timings_file");

    _fileName                                   = "timings.txt";
    const std::vector<std::string> lineElements = {
        "timings_file",
        "=",
        _fileName
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getTimingsFileName(), _fileName);
}

/**
 * @brief tests parsing the "rpmd_traj_file" command
 *
 */
TEST_F(TestInputFileReader, testRPMDTrajectoryFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rpmd_traj_file"));
    const auto &parseFunc = funcMap.at("rpmd_traj_file");

    _fileName                                   = "rpmd_traj.xyz";
    const std::vector<std::string> lineElements = {
        "rpmd_traj_file",
        "=",
        _fileName
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getRPMDTrajFileName(), _fileName);
}

/**
 * @brief tests parsing the "rpmd_restart_file" command
 *
 */
TEST_F(TestInputFileReader, testRPMDRestartFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rpmd_restart_file"));
    const auto &parseFunc = funcMap.at("rpmd_restart_file");

    _fileName                                   = "rpmd_traj.xyz";
    const std::vector<std::string> lineElements = {
        "rpmd_restart_file",
        "=",
        _fileName
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::OutputFileSettings::getRPMDRestartFileName(),
        _fileName
    );
}

/**
 * @brief tests parsing the "rpmd_energy_file" command
 *
 */
TEST_F(TestInputFileReader, testRPMDEnergyFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rpmd_energy_file"));
    const auto &parseFunc = funcMap.at("rpmd_energy_file");

    _fileName                                   = "rpmd_energy.txt";
    const std::vector<std::string> lineElements = {
        "rpmd_energy_file",
        "=",
        _fileName
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getRPMDEnergyFileName(), _fileName);
}

/**
 * @brief tests parsing the "rpmd_force_file" command
 *
 */
TEST_F(TestInputFileReader, testRPMDForceFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rpmd_force_file"));
    const auto &parseFunc = funcMap.at("rpmd_force_file");

    _fileName                                   = "rpmd_force.xyz";
    const std::vector<std::string> lineElements = {
        "rpmd_force_file",
        "=",
        _fileName
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getRPMDForceFileName(), _fileName);
}

/**
 * @brief tests parsing the "rpmd_charge_file" command
 *
 */
TEST_F(TestInputFileReader, testRPMDChargeFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rpmd_charge_file"));
    const auto &parseFunc = funcMap.at("rpmd_charge_file");

    _fileName                                   = "rpmd_charge.xyz";
    const std::vector<std::string> lineElements = {
        "rpmd_charge_file",
        "=",
        _fileName
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getRPMDChargeFileName(), _fileName);
}

/**
 * @brief tests parsing the "rpmd_velocity_file" command
 *
 */
TEST_F(TestInputFileReader, testRPMDVelocityFilename)
{
    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rpmd_vel_file"));
    const auto &parseFunc = funcMap.at("rpmd_vel_file");

    _fileName                                   = "rpmd_velocity.xyz";
    const std::vector<std::string> lineElements = {
        "rpmd_vel_file",
        "=",
        _fileName
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::OutputFileSettings::getRPMDVelocityFileName(),
        _fileName
    );
}

/**
 * @brief tests parsing the "overwrite_output" command
 *
 */
TEST_F(TestInputFileReader, parseOverwriteOutput)
{
    EXPECT_FALSE(settings::OutputFileSettings::getOverwriteOutputFiles());

    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("overwrite_output"));
    const auto &parseFunc = funcMap.at("overwrite_output");

    parseFunc({"overwrite_output", "=", "true"}, 0);
    EXPECT_TRUE(settings::OutputFileSettings::getOverwriteOutputFiles());

    _clearParser(parser);

    parseFunc({"overwrite_output", "=", "yes"}, 0);
    EXPECT_TRUE(settings::OutputFileSettings::getOverwriteOutputFiles());

    _clearParser(parser);

    parseFunc({"overwrite_output", "=", "on"}, 0);
    EXPECT_TRUE(settings::OutputFileSettings::getOverwriteOutputFiles());

    _clearParser(parser);

    parseFunc({"overwrite_output", "=", "false"}, 0);
    EXPECT_FALSE(settings::OutputFileSettings::getOverwriteOutputFiles());

    _clearParser(parser);

    parseFunc({"overwrite_output", "=", "no"}, 0);
    EXPECT_FALSE(settings::OutputFileSettings::getOverwriteOutputFiles());

    _clearParser(parser);

    parseFunc({"overwrite_output", "=", "off"}, 0);
    EXPECT_FALSE(settings::OutputFileSettings::getOverwriteOutputFiles());

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"overwrite_output", "=", "notABool"}, 0),
        exc::InputFileException,
        "Invalid value \"notABool\" for key \"overwrite_output\" at line 0 in "
        "input file. Allowed values: on|off|true|false|yes|no"
    )
}

/**
 * @brief tests parsing the "include_output_metadata" command
 *
 */
TEST_F(TestInputFileReader, parseIncludeOutputMetadata)
{
    EXPECT_FALSE(settings::OutputFileSettings::getIncludeOutputMetadata());

    input::OutputInputParser parser;
    const auto               funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("include_output_metadata"));
    const auto &parseFunc = funcMap.at("include_output_metadata");

    parseFunc({"include_output_metadata", "=", "true"}, 0);
    EXPECT_TRUE(settings::OutputFileSettings::getIncludeOutputMetadata());

    _clearParser(parser);

    parseFunc({"include_output_metadata", "=", "false"}, 0);
    EXPECT_FALSE(settings::OutputFileSettings::getIncludeOutputMetadata());

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"include_output_metadata", "=", "notABool"}, 0),
        exc::InputFileException,
        "Invalid value \"notABool\" for key \"include_output_metadata\" at "
        "line 0 in input file. Allowed values: on|off|true|false|yes|no"
    )
}
