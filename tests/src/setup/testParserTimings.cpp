#include "testInputFileReader.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace Setup::InputFileReader;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseTimestep)
{
    vector<string> lineElements = {"timestep", "=", "1"};
    _inputFileReader->parseTimestep(lineElements);
    EXPECT_EQ(_engine._timings.getTimestep(), 1.0);
}

TEST_F(TestInputFileReader, testParseNumberOfSteps)
{
    vector<string> lineElements = {"nsteps", "=", "1000"};
    _inputFileReader->parseNumberOfSteps(lineElements);
    EXPECT_EQ(_engine._timings.getNumberOfSteps(), 1000);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}