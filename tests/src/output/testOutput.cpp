#include "exceptions.hpp"         // for InputFileException
#include "output.hpp"             // for Output
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Test, Message, TestPartResult, InitG...
#include <cstdint>         // for UINT64_MAX
#include <memory>          // for allocator

/**
 * @brief tests setting output filename
 *
 */
TEST(TestOutput, testSpecialSetFilename)
{
    auto output = output::Output("default.out");
    EXPECT_THROW_MSG(output.setFilename(""), customException::InputFileException, "Filename cannot be empty");
    EXPECT_THROW_MSG(output.setFilename("src"), customException::InputFileException, "File already exists - filename = src");
}

/**
 * @brief tests setting output frequency
 *
 */
TEST(TestOutput, setSpecialOutputFrequency)
{
    auto output = output::Output("default.out");
    output::Output::setOutputFrequency(0);
    EXPECT_EQ(output.getOutputFrequency(), UINT64_MAX);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}