#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace setup;
using namespace ::testing;
using namespace output;

TEST_F(TestInputFileReader, testParseOutputFreq)
{
    vector<string> lineElements = {"outputfreq", "=", "1000"};
    _inputFileReader->parseOutputFreq(lineElements);
    EXPECT_EQ(Output::getOutputFrequency(), 1000);
}

TEST_F(TestInputFileReader, testParseLogFilename)
{
    _filename                   = "log.txt";
    vector<string> lineElements = {"logfilename", "=", _filename};
    _inputFileReader->parseLogFilename(lineElements);
    EXPECT_EQ(_engine._logOutput->getFilename(), _filename);
}

TEST_F(TestInputFileReader, testParseInfoFilename)
{
    _filename                   = "info.txt";
    vector<string> lineElements = {"infofilename", "=", "info.txt"};
    _inputFileReader->parseInfoFilename(lineElements);
    EXPECT_EQ(_engine._infoOutput->getFilename(), "info.txt");
}

TEST_F(TestInputFileReader, testParseEnergyFilename)
{
    _filename                   = "energy.txt";
    vector<string> lineElements = {"energyfilename", "=", _filename};
    _inputFileReader->parseEnergyFilename(lineElements);
    EXPECT_EQ(_engine._energyOutput->getFilename(), _filename);
}

TEST_F(TestInputFileReader, testParseTrajectoryFilename)
{
    _filename                   = "trajectory.xyz";
    vector<string> lineElements = {"trajectoryfilename", "=", _filename};
    _inputFileReader->parseTrajectoryFilename(lineElements);
    EXPECT_EQ(_engine._xyzOutput->getFilename(), _filename);
}

TEST_F(TestInputFileReader, testVelocityFilename)
{
    _filename                   = "velocity.xyz";
    vector<string> lineElements = {"velocityfilename", "=", _filename};
    _inputFileReader->parseVelocityFilename(lineElements);
    EXPECT_EQ(_engine._velOutput->getFilename(), _filename);
}

TEST_F(TestInputFileReader, testForceFilename)
{
    _filename                   = "force.xyz";
    vector<string> lineElements = {"forcefilename", "=", _filename};
    _inputFileReader->parseForceFilename(lineElements);
    EXPECT_EQ(_engine._forceOutput->getFilename(), _filename);
}

TEST_F(TestInputFileReader, testParseRestartFilename)
{
    _filename                   = "restart.xyz";
    vector<string> lineElements = {"restartfilename", "=", _filename};
    _inputFileReader->parseRestartFilename(lineElements);
    EXPECT_EQ(_engine._rstFileOutput->getFilename(), _filename);
}

TEST_F(TestInputFileReader, testChargeFilename)
{
    _filename                   = "charge.xyz";
    vector<string> lineElements = {"chargefilename", "=", _filename};
    _inputFileReader->parseChargeFilename(lineElements);
    EXPECT_EQ(_engine._chargeOutput->getFilename(), _filename);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}