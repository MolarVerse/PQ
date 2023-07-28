#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace setup;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseVirial)
{
    _engine.getVirial().getVirialType() = "molecular";
    vector<string> lineElements         = {"virial", "=", "atomic"};
    _inputFileReader->parseVirial(lineElements);
    EXPECT_EQ(_engine.getVirial().getVirialType(), "atomic");
    lineElements = {"virial", "=", "molecular"};
    _inputFileReader->parseVirial(lineElements);
    EXPECT_EQ(_engine.getVirial().getVirialType(), "molecular");
    lineElements = {"virial", "=", "notvalid"};
    EXPECT_THROW(_inputFileReader->parseVirial(lineElements), customException::InputFileException);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}