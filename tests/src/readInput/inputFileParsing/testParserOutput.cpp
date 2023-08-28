#include "energyOutput.hpp"            // for EnergyOutput
#include "engine.hpp"                  // for Engine
#include "exceptions.hpp"              // for InputFileException
#include "infoOutput.hpp"              // for InfoOutput
#include "inputFileParser.hpp"         // for readInput
#include "inputFileParserOutput.hpp"   // for InputFileParserOutput
#include "logOutput.hpp"               // for LogOutput
#include "output.hpp"                  // for Output, output
#include "rstFileOutput.hpp"           // for RstFileOutput
#include "testInputFileReader.hpp"     // for TestInputFileReader
#include "throwWithMessage.hpp"        // for EXPECT_THROW_MSG
#include "trajectoryOutput.hpp"        // for TrajectoryOutput

#include "gtest/gtest.h"   // for Message, TestPartResult, testing
#include <gtest/gtest.h>   // for TestInfo (ptr only), EXPECT_EQ
#include <iosfwd>          // for std
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace std;
using namespace readInput;
using namespace ::testing;
using namespace output;

/**
 * @brief tests parsing the "outputfreq" command
 *
 * @details if the outputfreq is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseOutputFreq)
{
    InputFileParserOutput parser(_engine);
    vector<string>        lineElements = {"outputfreq", "=", "1000"};
    parser.parseOutputFreq(lineElements, 0);
    EXPECT_EQ(Output::getOutputFrequency(), 1000);

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
    InputFileParserOutput parser(_engine);
    _fileName                   = "log.txt";
    vector<string> lineElements = {"logfilename", "=", _fileName};
    parser.parseLogFilename(lineElements, 0);
    EXPECT_EQ(_engine.getLogOutput().getFilename(), _fileName);
}

/**
 * @brief tests parsing the "info_file" command
 *
 */
TEST_F(TestInputFileReader, testParseInfoFilename)
{
    InputFileParserOutput parser(_engine);
    _fileName                   = "info.txt";
    vector<string> lineElements = {"infoFilename", "=", "info.txt"};
    parser.parseInfoFilename(lineElements, 0);
    EXPECT_EQ(_engine.getInfoOutput().getFilename(), "info.txt");
}

/**
 * @brief tests parsing the "energy_file" command
 *
 */
TEST_F(TestInputFileReader, testParseEnergyFilename)
{
    InputFileParserOutput parser(_engine);
    _fileName                   = "energy.txt";
    vector<string> lineElements = {"energyFilename", "=", _fileName};
    parser.parseEnergyFilename(lineElements, 0);
    EXPECT_EQ(_engine.getEnergyOutput().getFilename(), _fileName);
}

/**
 * @brief tests parsing the "traj_file" command
 *
 */
TEST_F(TestInputFileReader, testParseTrajectoryFilename)
{
    InputFileParserOutput parser(_engine);
    _fileName                   = "trajectory.xyz";
    vector<string> lineElements = {"trajectoryFilename", "=", _fileName};
    parser.parseTrajectoryFilename(lineElements, 0);
    EXPECT_EQ(_engine.getXyzOutput().getFilename(), _fileName);
}

/**
 * @brief tests parsing the "velocity_file" command
 *
 */
TEST_F(TestInputFileReader, testVelocityFilename)
{
    InputFileParserOutput parser(_engine);
    _fileName                   = "velocity.xyz";
    vector<string> lineElements = {"velocityFilename", "=", _fileName};
    parser.parseVelocityFilename(lineElements, 0);
    EXPECT_EQ(_engine.getVelOutput().getFilename(), _fileName);
}

/**
 * @brief tests parsing the "force_file" command
 *
 */
TEST_F(TestInputFileReader, testForceFilename)
{
    InputFileParserOutput parser(_engine);
    _fileName                   = "force.xyz";
    vector<string> lineElements = {"forceFilename", "=", _fileName};
    parser.parseForceFilename(lineElements, 0);
    EXPECT_EQ(_engine.getForceOutput().getFilename(), _fileName);
}

/**
 * @brief tests parsing the "restart_file" command
 *
 */
TEST_F(TestInputFileReader, testParseRestartFilename)
{
    InputFileParserOutput parser(_engine);
    _fileName                   = "restart.xyz";
    vector<string> lineElements = {"restartFilename", "=", _fileName};
    parser.parseRestartFilename(lineElements, 0);
    EXPECT_EQ(_engine.getRstFileOutput().getFilename(), _fileName);
}

/**
 * @brief tests parsing the "charge_file" command
 *
 */
TEST_F(TestInputFileReader, testChargeFilename)
{
    InputFileParserOutput parser(_engine);
    _fileName                   = "charge.xyz";
    vector<string> lineElements = {"chargeFilename", "=", _fileName};
    parser.parseChargeFilename(lineElements, 0);
    EXPECT_EQ(_engine.getChargeOutput().getFilename(), _fileName);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}