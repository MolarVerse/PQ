#include "outputFileSettings.hpp"

#include <gtest/gtest.h>   // for Message, TestPartResult, InitGoogleTest, Test

/**
 * @brief tests setting output frequency
 *
 */
TEST(TestOutputSettings, setSpecialOutputFrequency)
{
    settings::OutputFileSettings::setOutputFrequency(0);
    EXPECT_EQ(settings::OutputFileSettings::getOutputFrequency(), UINT64_MAX);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}