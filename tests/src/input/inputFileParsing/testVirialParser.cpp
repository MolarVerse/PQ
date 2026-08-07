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

#include <gtest/gtest.h>   // for InitGoogleTest, RUN_ALL_TESTS

#include <iosfwd>   // for std
#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "engine.hpp"                // for Engine
#include "exceptions.hpp"            // for InputFileException
#include "gtest/gtest.h"             // for Message, TestPartResult, testing
#include "inputFileParser.hpp"       // for readInput
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for EXPECT_THROW_MSG
#include "virial.hpp"                // for Virial
#include "virialInputParser.hpp"

using namespace std;
using namespace input;
using namespace ::testing;

/**
 * @brief tests parsing the "virial" command
 *
 * @details possible options are atomic or molecular - otherwise throws
 * inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseVirial)
{
    VirialInputParser        parser(*_engine);
    std::vector<std::string> lineElements = {"virial", "=", "atomic"};
    parser.parseVirial(lineElements, 0);
    EXPECT_EQ(_engine->getVirial().getVirialType(), "atomic");

    lineElements = {"virial", "=", "molecular"};
    parser.parseVirial(lineElements, 0);
    EXPECT_EQ(_engine->getVirial().getVirialType(), "molecular");

    lineElements = {"virial", "=", "notValid"};
    EXPECT_THROW_MSG(
        parser.parseVirial(lineElements, 0),
        customException::InputFileException,
        "Invalid virial setting \"notValid\" at line 0 in input file.\n"
        "Possible options are: molecular or atomic"
    );
}