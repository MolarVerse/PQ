#include "testInputFileReader.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace Setup::InputFileReader;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseJobType)
{
    vector<string> lineElements = {"jobtype", "=", "mm-md"};
    _inputFileReader->parseJobType(lineElements);
    EXPECT_EQ(_engine._jobType.getJobType(), "MMMD");
}

TEST_F(TestInputFileReader, testJobTypeUnknown)
{
    vector<string> lineElements = {"jobtype", "=", "unknown"};
    ASSERT_THROW(_inputFileReader->parseJobType(lineElements), InputFileException);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}