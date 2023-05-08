#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <string>
#include <cassert>

#include "testRstFileSection.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace testing;

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
    GTEST_SKIP();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
