#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <string>
#include <cassert>

#include "testRstFileSection.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace testing;

namespace Setup::RstFileReader
{
    TEST_F(TestNoseHooverSection, testKeyword)
    {
        EXPECT_EQ(_section->keyword(), "chi");
    }

    TEST_F(TestNoseHooverSection, testIsHeader)
    {
        EXPECT_TRUE(_section->isHeader());
    }

    TEST_F(TestNoseHooverSection, testNumberOfArguments)
    {
        GTEST_SKIP();
    }

    TEST_F(TestNoseHooverSection, testProcess)
    {
        auto line = vector<string>(0);
        ASSERT_THROW(_section->process(line, _engine), RstFileException);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
