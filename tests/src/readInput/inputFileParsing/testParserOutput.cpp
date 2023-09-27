/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "exceptions.hpp"              // for InputFileException
#include "inputFileParser.hpp"         // for readInput
#include "inputFileParserOutput.hpp"   // for InputFileParserOutput
#include "outputFileSettings.hpp"      // for OutputFileSettings
#include "testInputFileReader.hpp"     // for TestInputFileReader
#include "throwWithMessage.hpp"        // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for TestInfo (ptr only), EXPECT_EQ
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace readInput;

/**
 * @brief tests parsing the "outputfreq" command
 *
 * @details if the outputfreq is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseOutputFreq)
{
    InputFileParserOutput    parser(*_engine);
    std::vector<std::string> lineElements = {"outputfreq", "=", "1000"};
    parser.parseOutputFreq(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getOutputFrequency(), 1000);

    lineElements = {"outputfreq", "=", "-1000"};
    EXPECT_THROW_MSG(parser.parseOutputFreq(lineElements, 0),
                     customException::InputFileException,
                     "Output frequency cannot be negative - \"-1000\" at line 0 in input file");
}

/**
 * @brief tests parsing the "output_file" command
 *
 */
TEST_F(TestInputFileReader, testParseLogFilename)
{
    InputFileParserOutput parser(*_engine);
    _fileName                             = "log.txt";
    std::vector<std::string> lineElements = {"logfilename", "=", _fileName};
    parser.parseLogFilename(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getLogFileName(), _fileName);
}

/**
 * @brief tests parsing the "info_file" command
 *
 */
TEST_F(TestInputFileReader, testParseInfoFilename)
{
    InputFileParserOutput parser(*_engine);
    _fileName                             = "info.txt";
    std::vector<std::string> lineElements = {"infoFilename", "=", "info.txt"};
    parser.parseInfoFilename(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getInfoFileName(), "info.txt");
}

/**
 * @brief tests parsing the "energy_file" command
 *
 */
TEST_F(TestInputFileReader, testParseEnergyFilename)
{
    InputFileParserOutput parser(*_engine);
    _fileName                             = "energy.txt";
    std::vector<std::string> lineElements = {"energyFilename", "=", _fileName};
    parser.parseEnergyFilename(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getEnergyFileName(), _fileName);
}

/**
 * @brief tests parsing the "traj_file" command
 *
 */
TEST_F(TestInputFileReader, testParseTrajectoryFilename)
{
    InputFileParserOutput parser(*_engine);
    _fileName                             = "trajectory.xyz";
    std::vector<std::string> lineElements = {"trajectoryFilename", "=", _fileName};
    parser.parseTrajectoryFilename(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getTrajectoryFileName(), _fileName);
}

/**
 * @brief tests parsing the "velocity_file" command
 *
 */
TEST_F(TestInputFileReader, testVelocityFilename)
{
    InputFileParserOutput parser(*_engine);
    _fileName                             = "velocity.xyz";
    std::vector<std::string> lineElements = {"velocityFilename", "=", _fileName};
    parser.parseVelocityFilename(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getVelocityFileName(), _fileName);
}

/**
 * @brief tests parsing the "force_file" command
 *
 */
TEST_F(TestInputFileReader, testForceFilename)
{
    InputFileParserOutput parser(*_engine);
    _fileName                             = "force.xyz";
    std::vector<std::string> lineElements = {"forceFilename", "=", _fileName};
    parser.parseForceFilename(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getForceFileName(), _fileName);
}

/**
 * @brief tests parsing the "restart_file" command
 *
 */
TEST_F(TestInputFileReader, testParseRestartFilename)
{
    InputFileParserOutput parser(*_engine);
    _fileName                             = "restart.xyz";
    std::vector<std::string> lineElements = {"restartFilename", "=", _fileName};
    parser.parseRestartFilename(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getRestartFileName(), _fileName);
}

/**
 * @brief tests parsing the "charge_file" command
 *
 */
TEST_F(TestInputFileReader, testChargeFilename)
{
    InputFileParserOutput parser(*_engine);
    _fileName                             = "charge.xyz";
    std::vector<std::string> lineElements = {"chargeFilename", "=", _fileName};
    parser.parseChargeFilename(lineElements, 0);
    EXPECT_EQ(settings::OutputFileSettings::getChargeFileName(), _fileName);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}