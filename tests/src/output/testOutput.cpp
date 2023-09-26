#include "exceptions.hpp"         // for InputFileException
#include "output.hpp"             // for Output
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Test, Message, TestPartResult, InitGoogleTest, RUN_ALL_TESTS
#include <format>          // for format
#include <string>          // for string

/**
 * @brief tests setting output filename
 *
 */
TEST(TestOutput, testSpecialSetFilename)
{
    auto output = output::Output("default.out");
    EXPECT_THROW_MSG(output.setFilename(""), customException::InputFileException, "Filename cannot be empty");
    EXPECT_THROW_MSG(output.setFilename("src"), customException::InputFileException, "File already exists - filename = src");

    EXPECT_THROW_MSG(output.openFile(), customException::InputFileException, std::format("Could not open file - filename = src"));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}