#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;
using namespace output;

TEST_F(TestInputFileReader, testParseOutputFreq)
{
    vector<string> lineElements = {"outputfreq", "=", "1000"};
    _inputFileReader->parseOutputFreq(lineElements);
    EXPECT_EQ(Output::getOutputFrequency(), 1000);

    lineElements = {"outputfreq", "=", "-1000"};
    EXPECT_THROW(_inputFileReader->parseOutputFreq(lineElements), customException::InputFileException);
}

TEST_F(TestInputFileReader, testParseLogFilename)
{
    _filename                   = "log.txt";
    vector<string> lineElements = {"logfilename", "=", _filename};
    _inputFileReader->parseLogFilename(lineElements);
    EXPECT_EQ(_engine.getLogOutput().getFilename(), _filename);
}

TEST_F(TestInputFileReader, testParseInfoFilename)
{
    _filename                   = "info.txt";
    vector<string> lineElements = {"infoFilename", "=", "info.txt"};
    _inputFileReader->parseInfoFilename(lineElements);
    EXPECT_EQ(_engine.getInfoOutput().getFilename(), "info.txt");
}

TEST_F(TestInputFileReader, testParseEnergyFilename)
{
    _filename                   = "energy.txt";
    vector<string> lineElements = {"energyFilename", "=", _filename};
    _inputFileReader->parseEnergyFilename(lineElements);
    EXPECT_EQ(_engine.getEnergyOutput().getFilename(), _filename);
}

TEST_F(TestInputFileReader, testParseTrajectoryFilename)
{
    _filename                   = "trajectory.xyz";
    vector<string> lineElements = {"trajectoryFilename", "=", _filename};
    _inputFileReader->parseTrajectoryFilename(lineElements);
    EXPECT_EQ(_engine.getXyzOutput().getFilename(), _filename);
}

TEST_F(TestInputFileReader, testVelocityFilename)
{
    _filename                   = "velocity.xyz";
    vector<string> lineElements = {"velocityFilename", "=", _filename};
    _inputFileReader->parseVelocityFilename(lineElements);
    EXPECT_EQ(_engine.getVelOutput().getFilename(), _filename);
}

TEST_F(TestInputFileReader, testForceFilename)
{
    _filename                   = "force.xyz";
    vector<string> lineElements = {"forceFilename", "=", _filename};
    _inputFileReader->parseForceFilename(lineElements);
    EXPECT_EQ(_engine.getForceOutput().getFilename(), _filename);
}

TEST_F(TestInputFileReader, testParseRestartFilename)
{
    _filename                   = "restart.xyz";
    vector<string> lineElements = {"restartFilename", "=", _filename};
    _inputFileReader->parseRestartFilename(lineElements);
    EXPECT_EQ(_engine.getRstFileOutput().getFilename(), _filename);
}

TEST_F(TestInputFileReader, testChargeFilename)
{
    _filename                   = "charge.xyz";
    vector<string> lineElements = {"chargeFilename", "=", _filename};
    _inputFileReader->parseChargeFilename(lineElements);
    EXPECT_EQ(_engine.getChargeOutput().getFilename(), _filename);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}