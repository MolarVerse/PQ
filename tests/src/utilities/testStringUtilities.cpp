#include "exceptions.hpp"
#include "stringUtilities.hpp"

#include <fstream>
#include <gtest/gtest.h>
#include <string>

using namespace ::testing;

/**
 * @brief removeComments test by comment character
 *
 */
TEST(TestStringUtilities, removeComments)
{
    std::string line        = "test;test";
    std::string commentChar = ";";
    std::string result      = "test";
    EXPECT_EQ(result, StringUtilities::removeComments(line, commentChar));

    std::string line2 = ";test";
    EXPECT_TRUE(StringUtilities::removeComments(line2, commentChar).empty());
}

/**
 * @brief getLineCommands separated by semicolon
 *
 */
TEST(TestStringUtilities, getLineCommands)
{
    std::string line = "test;test2;";
    EXPECT_EQ(2, StringUtilities::getLineCommands(line, 0).size() - 1);
    EXPECT_EQ("test", StringUtilities::getLineCommands(line, 0)[0]);
    EXPECT_EQ("test2", StringUtilities::getLineCommands(line, 0)[1]);

    std::string line2 = "test";
    EXPECT_THROW(StringUtilities::getLineCommands(line2, 0), customException::InputFileException);
}

/**
 * @brief test splitString by whitespace
 *
 */
TEST(TestStringUtilities, splitString)
{
    std::string line = "test test2";
    EXPECT_EQ(2, StringUtilities::splitString(line).size());
    EXPECT_EQ("test", StringUtilities::splitString(line)[0]);
    EXPECT_EQ("test2", StringUtilities::splitString(line)[1]);
}

/**
 * @brief test toLowerCopy function
 *
 */
TEST(TestStringUtilities, toLowerCopy)
{
    std::string line = "TEST";
    EXPECT_EQ("test", StringUtilities::toLowerCopy(line));
}

/**
 * @brief test check if file exists
 *
 */
TEST(TestStringUtilities, fileExists)
{
    std::string   file = "testFile.txt";
    std::ofstream out(file);
    out.close();
    EXPECT_TRUE(StringUtilities::fileExists(file));
    EXPECT_FALSE(StringUtilities::fileExists("testFile2.txt"));
    std::remove(file.c_str());
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}