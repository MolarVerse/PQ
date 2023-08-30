#include "engine.hpp"                   // for Engine
#include "exceptions.hpp"               // for RstFileException, customException
#include "restartFileSection.hpp"       // for RstFileSection, readInput
#include "testRestartFileSection.hpp"   // for TestStepCountSection
#include "timings.hpp"                  // for Timings

#include "gtest/gtest.h"   // for AssertionResult, Message, TestPart...
#include <cstddef>         // for size_t
#include <gtest/gtest.h>   // for TestInfo (ptr only), TEST_F, InitG...
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace readInput;

TEST_F(TestStepCountSection, testKeyword) { EXPECT_EQ(_section->keyword(), "step"); }

TEST_F(TestStepCountSection, testIsHeader) { EXPECT_TRUE(_section->isHeader()); }

TEST_F(TestStepCountSection, testNumberOfArguments)
{
    for (size_t i = 0; i < 10; ++i)
        if (i != 2)
        {
            auto line = std::vector<std::string>(i);
            ASSERT_THROW(_section->process(line, _engine), customException::RstFileException);
        }
}

TEST_F(TestStepCountSection, testNegativeStepCount)
{
    auto line = std::vector<std::string>(2);
    line[1]   = "-1";
    ASSERT_THROW(_section->process(line, _engine), customException::RstFileException);
}

TEST_F(TestStepCountSection, testProcess)
{
    auto line = std::vector<std::string>(2);
    line[1]   = "1000";
    _section->process(line, _engine);
    EXPECT_EQ(_engine.getTimings().getStepCount(), 1000);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}