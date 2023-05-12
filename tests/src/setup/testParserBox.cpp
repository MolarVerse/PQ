#include "testInputFileReader.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace Setup::InputFileReader;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseDensity)
{
    vector<string> lineElements = {"density", "=", "1.0"};
    _inputFileReader->parseDensity(lineElements);
    EXPECT_EQ(_engine.getSimulationBox()._box.getDensity(), 1.0);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}