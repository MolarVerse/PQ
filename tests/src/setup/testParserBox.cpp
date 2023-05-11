#include "testInputFileReader.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace Setup::InputFileReader;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseDensity)
{
    vector<string> lineElements = {"density", "=", "1.0"};
    _inputFileReader->parseDensity(lineElements);
    EXPECT_EQ(_engine._simulationBox._box.getDensity(), 1.0);
}

TEST_F(TestInputFileReader, testParseNumberOfArguments)
{
    for (int i = 0; i < 10; i++)
    {
        if (i != 3)
        {
            auto lineElements = vector<string>(i);
            ASSERT_THROW(_inputFileReader->parseDensity(lineElements), InputFileException);
        }
    }
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}