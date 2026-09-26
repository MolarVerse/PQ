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
#include "inputKeyAdapter.hpp"
#include "inputParam.hpp"
#include "throwWithMessage.hpp"

using namespace input;

/**
 * @brief tests that calling the adapted ParseFunc behaves identically to
 * calling InputKey<T>::parse directly
 *
 */
TEST(TestInputKeyAdapter, adaptedParseFuncMatchesDirectParse)
{
    double captured = 0.0;

    InputKey<double> key(
        KeyRegistry<double>{
            .metadata =
                KeyMetadata{
                    .name         = "timestep",
                    .title        = "Timestep",
                    .description  = "The timestep for the simulation",
                    .unit         = "fs",
                    .errorMessage = "Invalid timestep value",
                },
            .onSet = [&captured](const double &value) { captured = value; }
        }
    );

    const InputFileParser::ParseFunc parseFunc = adapt(key);
    parseFunc({"timestep", "=", "0.5"}, 1);

    EXPECT_TRUE(key.isSet());
    EXPECT_EQ(key.value(), 0.5);
    EXPECT_EQ(captured, 0.5);
}

/**
 * @brief tests that an exception thrown by InputKey<T>::parse propagates
 * unchanged through the adapted ParseFunc
 *
 */
TEST(TestInputKeyAdapter, exceptionsPropagateThroughAdapter)
{
    InputKey<double> key(
        KeyRegistry<double>{
            .metadata =
                KeyMetadata{
                    .name         = "timestep",
                    .title        = "Timestep",
                    .description  = "The timestep for the simulation",
                    .unit         = "fs",
                    .errorMessage = "Invalid timestep value",
                }
        }
    );

    const InputFileParser::ParseFunc parseFunc = adapt(key);

    EXPECT_THROW_MSG(
        parseFunc({"timestep", "=", "not_a_number"}, 3),
        exc::InputFileException,
        "Invalid timestep value at line 3 in input file"
    );
}

/**
 * @brief tests the adapter wired into the real InputFileParser::addKeyword
 * / dispatch mechanism, exactly as a migrated parser class will use it --
 * this is the actual integration surface the whole design depends on
 *
 */
TEST(TestInputKeyAdapter, wiresIntoRealInputFileParserAddKeyword)
{
    InputFileParser parser;
    InputRegistry   registry;

    auto &timestepKey = registry.registerKey<double>(KeyRegistry<double>{
        .metadata =
            KeyMetadata{
                .name         = "timestep",
                .title        = "Timestep",
                .description  = "The timestep for the simulation",
                .unit         = "fs",
                .errorMessage = "Invalid timestep value",
            },
        .defaultValue = 1.0
    });

    parser.addKeyword(
        "timestep",
        [&registry](
            const std::vector<std::string> &lineElements,
            const size_t                    lineNumber
        ) { registry.parseLine(lineElements, lineNumber); },
        /*required=*/false
    );

    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("timestep"));

    funcMap.at("timestep")({"timestep", "=", "2.5"}, 1);

    EXPECT_EQ(timestepKey.value(), 2.5);
}

/**
 * @brief tests adapt() used directly per-key (the documented pattern for
 * a single InputKey<T>, as opposed to routing a whole registry through
 * one ParseFunc as in the previous test)
 *
 */
TEST(TestInputKeyAdapter, adaptWiresSingleKeyIntoAddKeyword)
{
    InputFileParser parser;

    InputKey<bool> key(
        KeyRegistry<bool>{
            .metadata =
                KeyMetadata{
                    .name         = "verbose",
                    .title        = "Verbose",
                    .description  = "Enable verbose output",
                    .unit         = "",
                    .errorMessage = "Invalid verbose value",
                }
        }
    );

    parser.addKeyword("verbose", adapt(key), /*required=*/false);

    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("verbose"));

    funcMap.at("verbose")({"verbose", "=", "on"}, 1);

    EXPECT_TRUE(key.isSet());
    EXPECT_EQ(key.value(), true);
}
