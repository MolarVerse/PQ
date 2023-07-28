#include "testInputFileReader.hpp"

using namespace std;
using namespace readInput;
using namespace customException;

TEST(TestInputFileReader, testCheckCommand)
{
    for (int i = 0; i < 10; i++)
    {
        auto lineElements = vector<string>(i);

        if (i != 3)
        {
            ASSERT_THROW(checkCommand(lineElements, 1), InputFileException);
        }
        else
        {
            lineElements[1] = "=";
            ASSERT_NO_THROW(checkCommand(lineElements, 1));
        }
    }
}

TEST(TestInputFileReader, testCheckCommandArray)
{
    for (int i = 0; i < 10; i++)
    {
        auto lineElements = vector<string>(i);

        if (i < 3)
        {
            ASSERT_THROW(checkCommandArray(lineElements, 1), InputFileException);
        }
        else
        {
            lineElements[1] = "=";
            ASSERT_NO_THROW(checkCommandArray(lineElements, 1));
        }
    }
}

TEST(TestInputFileReader, testEqualSign)
{
    ASSERT_THROW(checkEqualSign("a", 1), InputFileException);

    ASSERT_NO_THROW(checkEqualSign("=", 1));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}