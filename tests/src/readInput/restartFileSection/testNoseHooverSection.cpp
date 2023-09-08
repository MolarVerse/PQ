#include "exceptions.hpp"               // for RstFileException, customException
#include "restartFileSection.hpp"       // for RstFileSection, readInput
#include "testRestartFileSection.hpp"   // for TestNoseHooverSection

#include "gtest/gtest.h"   // for AssertionResult, Message, TestPart...
#include <gtest/gtest.h>   // for TestInfo (ptr only), TEST_F, InitG...
#include <string>          // for string, allocator
#include <vector>          // for vector

using namespace readInput;

TEST_F(TestNoseHooverSection, testKeyword) { EXPECT_EQ(_section->keyword(), "chi"); }

TEST_F(TestNoseHooverSection, testIsHeader) { EXPECT_TRUE(_section->isHeader()); }

TEST_F(TestNoseHooverSection, testNumberOfArguments) { GTEST_SKIP(); }

TEST_F(TestNoseHooverSection, testProcess)
{
    auto line = std::vector<std::string>(0);
    ASSERT_THROW(_section->process(line, _engine), customException::RstFileException);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}
