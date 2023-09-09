#include "color.hpp"        // for Code
#include "exceptions.hpp"   // for CustomException

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for Test, CaptureStdout, GetCapturedStdout
#include <string>          // for allocator, string
#include <string_view>     // for string_view

/**
 * @brief tests colorful output for FG_RED
 *
 */
TEST(TestColor, redException)
{
    testing::internal::CaptureStdout();
    auto customException = customException::CustomException("test");
    customException.colorfulOutput(Color::FG_RED, "test");
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_STREQ(output.c_str(), "\033[31mtest\033[39m\n");
}

/**
 * @brief tests colorful output for FG_ORANGE
 *
 */
TEST(TestColor, orangeException)
{
    testing::internal::CaptureStdout();
    auto customException = customException::CustomException("test");
    customException.colorfulOutput(Color::FG_ORANGE, "test");
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_STREQ(output.c_str(), "\033[33mtest\033[39m\n");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}