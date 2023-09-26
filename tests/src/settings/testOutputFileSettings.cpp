#include "outputFileSettings.hpp"   // for OutputFileSettings

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for Test, InitGoogleTest, RUN_ALL_TESTS
#include <memory>          // for allocator
#include <stdint.h>        // for UINT64_MAX

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