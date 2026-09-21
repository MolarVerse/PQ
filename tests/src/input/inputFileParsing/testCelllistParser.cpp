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

#include <gtest/gtest.h>   // for TestInfo (ptr only), InitGoogleTest, RUN_ALL_TESTS, TEST_F, TestPartResult

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "cellListInputParser.hpp"   // for CellListInputParser
#include "engine.hpp"                // for Engine
#include "exceptions.hpp"            // for InputFileException
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for EXPECT_THROW_MSG

using namespace input;

/**
 * @brief tests parsing the "cell-list" command
 *
 * @details possible options are on or off - otherwise throws inputFileException
 *
 */
TEST_F(TestInputFileReader, parseCellListActivated)
{
    CellListInputParser parser(_engine->getCellList());
    const auto          funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("cell_list"));
    const auto& parseFunc = funcMap.at("cell_list");

    std::vector<std::string> lineElements = {"cell-list", "=", "off"};
    parseFunc(lineElements, 0);
    EXPECT_FALSE(settings::Settings::isCellListActivated());

    clearParser(parser);

    lineElements = {"cell-list", "=", "on"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::Settings::isCellListActivated());

    clearParser(parser);

    lineElements = {"cell-list", "=", "notValid"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"cell-list\" at line 0 "
        "in input file. Possible options are: on|off|true|false|yes|no"
    );
}

/**
 * @brief tests parsing the "cell-number" command
 *
 * @details if the number of cells is negative or 0, throws inputFileException
 *
 */
TEST_F(TestInputFileReader, numberOfCells)
{
    CellListInputParser parser(_engine->getCellList());
    const auto          funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("cell_number"));
    const auto& parseFunc = funcMap.at("cell_number");

    std::vector<std::string> lineElements = {"cell-number", "=", "3"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        _engine->getCellList()->getNumberOfCells(),
        linalg::Vec3Dul(3, 3, 3)
    );

    clearParser(parser);

    lineElements = {"cell-number", "=", "0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"cell-number\" at line 0 in input file: "
        "failed validation with message Value must be greater than or equal to "
        "1"
    );
}
