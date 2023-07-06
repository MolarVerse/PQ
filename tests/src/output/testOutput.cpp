#include "exceptions.hpp"
#include "output.hpp"

#include "gtest/gtest.h"
#include <filesystem>

TEST(TestOutput, TestSpecialSetFilename)
{
    auto output = output::Output("default.out");
    EXPECT_THROW(output.setFilename(""), customException::InputFileException);
    EXPECT_THROW(output.setFilename("src"), customException::InputFileException);
}

TEST(TestOutput, setSpecialOutputFrequency)
{
    auto output = output::Output("default.out");
    output::Output::setOutputFrequency(0);
    EXPECT_EQ(output.getOutputFrequency(), INT64_MAX);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}