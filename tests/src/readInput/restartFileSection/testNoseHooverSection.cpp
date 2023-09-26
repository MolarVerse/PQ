#include "restartFileSection.hpp"   // for RstFileSection, readInput

#include <gtest/gtest.h>   // for TestInfo (ptr only), TEST_F, InitG...

using namespace readInput;

// TEST_F(TestNoseHooverSection, testKeyword) { EXPECT_EQ(_section->keyword(), "chi"); }

// TEST_F(TestNoseHooverSection, testIsHeader) { EXPECT_TRUE(_section->isHeader()); }

// TEST_F(TestNoseHooverSection, testNumberOfArguments) { GTEST_SKIP(); }

// TEST_F(TestNoseHooverSection, testProcess)
// {
//     auto line = std::vector<std::string>(0);
//     ASSERT_THROW(_section->process(line, *_engine), customException::RstFileException);
// }

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}
