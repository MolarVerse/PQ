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