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

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "exceptions.hpp"
#include "inputFileParser.hpp"
#include "parserUtils.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

/**
 * @brief tests input::checkCommand function
 *
 * @details if the number of arguments is not 3 it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, checkCommand)
{
    auto lineElements = std::vector<std::string>{"test", "="};
    ASSERT_THROW_MSG(
        input::checkCommand(lineElements, 1),
        exc::InputFileException,
        "Invalid number of arguments at line 1 in input file"
    );

    lineElements = std::vector<std::string>{"test", "=", "test2", "tooMany"};
    ASSERT_THROW_MSG(
        input::checkCommand(lineElements, 1),
        exc::InputFileException,
        "Invalid number of arguments at line 1 in input file"
    );

    lineElements = std::vector<std::string>{"test", "=", "test2"};
    ASSERT_NO_THROW(input::checkCommand(lineElements, 1));
}

/**
 * @brief tests checkCommandArray function
 *
 * @details if the number of arguments is less than 3 it throws
 * inputFileException
 *
 */
TEST_F(TestInputFileReader, checkCommandArray)
{
    auto lineElements = std::vector<std::string>{"test", "="};
    ASSERT_THROW_MSG(
        input::checkCommandArray(lineElements, 1),
        exc::InputFileException,
        "Invalid number of arguments at line 1 in input file"
    );

    lineElements = std::vector<std::string>{"test", "=", "test2", "OK"};
    ASSERT_NO_THROW(input::checkCommandArray(lineElements, 1));

    lineElements = std::vector<std::string>{"test", "=", "test2"};
    ASSERT_NO_THROW(input::checkCommandArray(lineElements, 1));
}

/**
 * @brief tests input::checkEqualSign function
 *
 */
TEST_F(TestInputFileReader, equalSign)
{
    ASSERT_THROW_MSG(
        input::checkEqualSign("a", 1),
        exc::InputFileException,
        "Invalid command at line 1 in input file"
    );

    ASSERT_NO_THROW(input::checkEqualSign("=", 1));
}
