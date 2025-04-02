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

#include <filesystem>   // for filesystem
#include <format>       // for format
#include <string>       // for string

#include "exceptions.hpp"   // for InputFileException
#include "gtest/gtest.h"   // for Test, Message, TestPartResult, InitGoogleTest, RUN_ALL_TESTS
#include "output.hpp"               // for Output
#include "outputFileSettings.hpp"   // for OutputFileSettings
#include "throwWithMessage.hpp"     // for EXPECT_THROW_MSG

using namespace settings;

/**
 * @brief tests setting output filename
 *
 */
TEST(TestOutput, testSpecialSetFilename)
{
    auto output = output::Output("default.log");

    EXPECT_THROW_MSG(
        output.setFilename(""),
        customException::InputFileException,
        "Filename cannot be empty"
    );

    EXPECT_THROW_MSG(
        output.setFilename("src"),
        customException::InputFileException,
        "File already exists - filename = src"
    );

    EXPECT_THROW_MSG(
        output.openFile(),
        customException::InputFileException,
        std::format("Could not open file - filename = src")
    );

    const std::string testFileName = "test_output.txt";
    std::ofstream     testFile(testFileName);
    testFile.close();

    EXPECT_THROW_MSG(
        output.setFilename(testFileName),
        customException::InputFileException,
        std::format("File already exists - filename = {}", testFileName)
    );

    OutputFileSettings::setOverwriteOutputFiles(true);

    EXPECT_NO_THROW(output.setFilename(testFileName));
    EXPECT_NO_THROW(output.close());
    EXPECT_EQ(output.getFilename(), testFileName);

    OutputFileSettings::setOverwriteOutputFiles(false);
    std::filesystem::remove(testFileName);
}