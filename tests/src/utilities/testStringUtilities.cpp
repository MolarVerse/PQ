#include "exceptions.hpp"        // for InputFileException
#include "stringUtilities.hpp"   // for getLineCommands, splitString, fileExists

#include "gmock/gmock.h"   // for ElementsAre, MakePredicateFormatter
#include "gtest/gtest.h"   // for AssertionResult, Message, TestPartResult
#include <cstdio>          // for remove
#include <fstream>         // for ofstream
#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), EXPECT_EQ
#include <string>          // for string, allocator
#include <vector>          // for vector

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
    std::string line2 = "test";
    EXPECT_THROW(utilities::getLineCommands(line2, 0), customException::InputFileException);
    auto *line = "nstep = 1";
    ASSERT_THROW(utilities::getLineCommands(line, 1), customException::InputFileException);

    line = "nstep = 1;";
    ASSERT_THAT(utilities::getLineCommands(line, 1), testing::ElementsAre("nstep = 1"));

    line = "nstep = 1; nstep = 2";
    ASSERT_THROW(utilities::getLineCommands(line, 1), customException::InputFileException);

    line = "nstep = 1; nstep = 2;";
    ASSERT_THAT(utilities::getLineCommands(line, 1), testing::ElementsAre("nstep = 1", " nstep = 2"));
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
 * @brief test firstLetterToUpperCaseCopy function
 *
 */
TEST(TestStringUtilities, firstLetterToUpperCaseCopy)
{
    std::string line = "TEST";
    EXPECT_EQ("Test", utilities::firstLetterToUpperCaseCopy(line));

    line = "test";
    EXPECT_EQ("Test", utilities::firstLetterToUpperCaseCopy(line));
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
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}