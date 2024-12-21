/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), EXPECT_EQ

#include <cstdio>    // for remove
#include <fstream>   // for ofstream
#include <string>    // for string, allocator
#include <vector>    // for vector

#include "exceptions.hpp"        // for InputFileException
#include "gmock/gmock.h"         // for ElementsAre, MakePredicateFormatter
#include "gtest/gtest.h"         // for AssertionResult, Message, TestPartResult
#include "stringUtilities.hpp"   // for getLineCommands, splitString, fileExists

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
    EXPECT_THROW(
        utilities::getLineCommands(line2, 0),
        customException::InputFileException
    );
    auto *line = "nstep = 1";
    ASSERT_THROW(
        utilities::getLineCommands(line, 1),
        customException::InputFileException
    );

    line = "nstep = 1;";
    ASSERT_THAT(
        utilities::getLineCommands(line, 1),
        testing::ElementsAre("nstep = 1")
    );

    line = "nstep = 1; nstep = 2";
    ASSERT_THROW(
        utilities::getLineCommands(line, 1),
        customException::InputFileException
    );

    line = "nstep = 1; nstep = 2;";
    ASSERT_THAT(
        utilities::getLineCommands(line, 1),
        testing::ElementsAre("nstep = 1", " nstep = 2")
    );
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
 * @brief test toLowerAndReplaceDashesCopy function
 *
 */
TEST(TestStringUtilities, toLowerAndReplaceDashesCopy)
{
    std::string line = "TE-S--T";
    EXPECT_EQ("te_s__t", utilities::toLowerAndReplaceDashesCopy(line));
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