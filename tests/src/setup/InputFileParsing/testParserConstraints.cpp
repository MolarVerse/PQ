#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace setup;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseShakeActivated)
{
    vector<string> lineElements = {"shake", "=", "off"};
    _inputFileReader->parseShakeActivated(lineElements);
    EXPECT_FALSE(_engine.getConstraints().isActivated());
    lineElements = {"shake", "=", "on"};
    _inputFileReader->parseShakeActivated(lineElements);
    EXPECT_TRUE(_engine.getConstraints().isActivated());
    lineElements = {"shake", "=", "1"};
    EXPECT_THROW(_inputFileReader->parseShakeActivated(lineElements), customException::InputFileException);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}