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

#include <filesystem>
#include <format>
#include <string>

#include "exceptions.hpp"
#include "output.hpp"
#include "throwWithMessage.hpp"

struct TestOutputUtils
{
    static void openFile(out::Output &output) { output._openFile(); }
};

/**
 * @brief tests setting output filename
 *
 */
TEST(TestOutput, testSpecialSetFilename)
{
    auto output = out::Output("default.log");

    EXPECT_THROW_MSG(
        output.setFilename("", false),
        exc::InputFileException,
        "Filename cannot be empty"
    );

    const std::string testDirName = "test_output_dir";
    std::filesystem::create_directory(testDirName);

    EXPECT_THROW_MSG(
        output.setFilename(testDirName, false),
        exc::InputFileException,
        std::format("File already exists - filename = {}", testDirName)
    );

    EXPECT_THROW_MSG(
        TestOutputUtils::openFile(output),
        exc::InputFileException,
        std::format("Could not open file - filename = {}", testDirName)
    );

    std::filesystem::remove(testDirName);

    const std::string testFileName = "test_output.txt";
    std::ofstream     testFile(testFileName);
    testFile.close();

    EXPECT_THROW_MSG(
        output.setFilename(testFileName, false),
        exc::InputFileException,
        std::format("File already exists - filename = {}", testFileName)
    );

    EXPECT_NO_THROW(output.setFilename(testFileName, true));
    EXPECT_NO_THROW(output.close());
    EXPECT_EQ(output.getFilename(), testFileName);

    std::filesystem::remove(testFileName);
}
