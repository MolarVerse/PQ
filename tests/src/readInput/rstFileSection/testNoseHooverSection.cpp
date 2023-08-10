#include "exceptions.hpp"
#include "testRstFileSection.hpp"

#include <cassert>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace std;
using namespace testing;
using namespace readInput;
using namespace customException;

TEST_F(TestNoseHooverSection, testKeyword) { EXPECT_EQ(_section->keyword(), "chi"); }

TEST_F(TestNoseHooverSection, testIsHeader) { EXPECT_TRUE(_section->isHeader()); }

TEST_F(TestNoseHooverSection, testNumberOfArguments) { GTEST_SKIP(); }

TEST_F(TestNoseHooverSection, testProcess)
{
    auto line = vector<string>(0);
    ASSERT_THROW(_section->process(line, _engine), RstFileException);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
