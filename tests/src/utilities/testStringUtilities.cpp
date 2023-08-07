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
    EXPECT_EQ(result, utilities::removeComments(line, commentChar));

    std::string line2 = ";test";
    EXPECT_TRUE(utilities::removeComments(line2, commentChar).empty());
}

/**
 * @brief getLineCommands separated by semicolon
 *
 */
TEST(TestStringUtilities, getLineCommands)
{
    std::string line = "test;test2;";
    EXPECT_EQ(2, utilities::getLineCommands(line, 0).size() - 1);
    EXPECT_EQ("test", utilities::getLineCommands(line, 0)[0]);
    EXPECT_EQ("test2", utilities::getLineCommands(line, 0)[1]);

    std::string line2 = "test";
    EXPECT_THROW(utilities::getLineCommands(line2, 0), customException::InputFileException);
}

/**
 * @brief test splitString by whitespace
 *
 */
TEST(TestStringUtilities, splitString)
{
    std::string line = "test test2";
    EXPECT_EQ(2, utilities::splitString(line).size());
    EXPECT_EQ("test", utilities::splitString(line)[0]);
    EXPECT_EQ("test2", utilities::splitString(line)[1]);
}

/**
 * @brief test toLowerCopy function
 *
 */
TEST(TestStringUtilities, toLowerCopy)
{
    std::string line = "TEST";
    EXPECT_EQ("test", utilities::toLowerCopy(line));
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
    EXPECT_TRUE(utilities::fileExists(file));
    EXPECT_FALSE(utilities::fileExists("testFile2.txt"));
    std::remove(file.c_str());
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}