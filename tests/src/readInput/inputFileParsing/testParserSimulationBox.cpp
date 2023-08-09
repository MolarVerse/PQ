#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseDensity)
{
    const vector<string> lineElements = {"density", "=", "1.0"};
    _inputFileReader->parseDensity(lineElements, 0);
    EXPECT_EQ(_engine.getSimulationBox().getDensity(), 1.0);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}