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

TEST_F(TestStepCountSection, testKeyword) { EXPECT_EQ(_section->keyword(), "step"); }

TEST_F(TestStepCountSection, testIsHeader) { EXPECT_TRUE(_section->isHeader()); }

TEST_F(TestStepCountSection, testNumberOfArguments)
{
    for (int i = 0; i < 10; ++i)
        if (i != 2)
        {
            auto line = vector<string>(i);
            ASSERT_THROW(_section->process(line, _engine), RstFileException);
        }
}

TEST_F(TestStepCountSection, testProcess)
{
    auto line = vector<string>(2);
    line[1]   = "1000";
    _section->process(line, _engine);
    EXPECT_EQ(_engine.getTimings().getStepCount(), 1000);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}