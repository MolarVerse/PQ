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
#include "ringPolymerInputParser.hpp"
#include "ringPolymerSettings.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace input;

/**
 * @brief tests parsing the "rpmd_n_replica" command
 *
 * @details if the number of replicas is lower than 2 it throws
 * inputFileException. Dispatches through getKeywordFuncMap(), the same
 * path InputFileReader::process uses in production, since the migrated
 * parser no longer exposes parseNumberOfBeads as a standalone method --
 * parsing now lives in the registered InputKey<size_t> itself.
 *
 */
TEST_F(TestInputFileReader, testParseNumberOfReplicas)
{
    RingPolymerInputParser parser;
    const auto             funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rpmd_n_replica"));
    const auto &parseFunc = funcMap.at("rpmd_n_replica");

    std::vector<std::string> lineElements = {"rpmd_n_replica", "=", "10"};
    parseFunc(lineElements, 0);

    EXPECT_EQ(settings::RingPolymerSettings::getNumberOfBeads(), 10);

    clearParser(parser);

    lineElements = {"rpmd_n_replica", "=", "1"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"1\" for key \"rpmd_n_replica\" at line 0 in input "
        "file: failed validation with message Value must be greater than or "
        "equal to 2"
    );
}

/**
 * @brief tests that a genuinely malformed (non-numeric) value produces a
 * clean InputFileException rather than propagating a raw std::invalid_argument
 *
 * @details this is new coverage the original hand-written parseNumberOfBeads
 * didn't have: the old function let std::invalid_argument from
 * utilities::stringToInt propagate uncaught, relying on
 * InputFileReader::process's catch-all to wrap it. The migrated key
 * handles this itself via Converter<size_t>::tryParse, with an equivalent
 * but not byte-identical message (see PR description for the exact
 * wording tradeoff).
 *
 */
TEST_F(TestInputFileReader, testParseNumberOfReplicasInvalidSyntax)
{
    RingPolymerInputParser parser;
    const auto             funcMap   = parser.getKeywordFuncMap();
    const auto            &parseFunc = funcMap.at("rpmd_n_replica");

    const std::vector<std::string> lineElements = {
        "rpmd_n_replica",
        "=",
        "not_a_number"
    };

    EXPECT_THROW(parseFunc(lineElements, 0), exc::InputFileException);
}
