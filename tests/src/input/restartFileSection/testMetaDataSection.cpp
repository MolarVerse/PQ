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

#include <cstddef>
#include <string>
#include <vector>

#include "enums/rstMetaData.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "restartFileReader/restartFileSection.hpp"
#include "testRestartFileSection.hpp"
#include "throwWithMessage.hpp"
#include "version.hpp"

namespace
{
    const auto metaDataKeyword = std::string(RestartFileMetaDataKeyWord);
}   // namespace

/**
 * @brief the keyword is the lower case metadata marker, so that it matches the
 * normalised first element of a line in the restart file reader
 *
 */
TEST_F(TestMetaDataSection, keyword)
{
    EXPECT_EQ(_section->keyword(), "!meta_data");
}

/**
 * @brief a valid version line sets the restart version
 *
 */
TEST_F(TestMetaDataSection, processVersion)
{
    EXPECT_FALSE(isValid(_settings.generalSettings.getRestartVersion()));

    auto   line = std::vector<std::string>{"!meta_data", "version", "v2.1.3"};
    size_t lineNumber = 7;
    _section->process(line, *_engine, lineNumber);

    const auto version = _settings.generalSettings.getRestartVersion();
    EXPECT_TRUE(isValid(version));
    EXPECT_EQ(version, SemVer::from_string("v2.1.3"));
}

/**
 * @brief the key of the metadata line is case insensitive
 *
 */
TEST_F(TestMetaDataSection, processVersionCaseInsensitive)
{
    auto   line = std::vector<std::string>{"!meta_data", "VeRsIoN", "v1.2.0"};
    size_t lineNumber = 1;
    _section->process(line, *_engine, lineNumber);

    EXPECT_EQ(
        _settings.generalSettings.getRestartVersion(),
        SemVer::from_string("v1.2.0")
    );
}

/**
 * @brief the keyword of the line is matched case insensitively, as the writer
 * writes it in upper case
 *
 */
TEST_F(TestMetaDataSection, processUpperCaseKeyword)
{
    auto line = std::vector<std::string>{
        RestartFileMetaDataKeyWord,
        "Version",
        "v1.4.0"
    };
    size_t lineNumber = 1;
    _section->process(line, *_engine, lineNumber);

    EXPECT_EQ(
        _settings.generalSettings.getRestartVersion(),
        SemVer::from_string("v1.4.0")
    );
}

/**
 * @brief a version with a git hash extension is parsed
 *
 */
TEST_F(TestMetaDataSection, processVersionWithHash)
{
    auto line =
        std::vector<std::string>{"!meta_data", "version", "v2.0.0+abc123"};
    size_t lineNumber = 1;
    _section->process(line, *_engine, lineNumber);

    EXPECT_EQ(
        _settings.generalSettings.getRestartVersion(),
        SemVer::from_string("v2.0.0+abc123")
    );
}

/**
 * @brief the section throws if the first element is not the keyword
 *
 */
TEST_F(TestMetaDataSection, wrongKeyword)
{
    size_t lineNumber = 3;

    auto line = std::vector<std::string>{"box", "version", "v1.0.0"};
    ASSERT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "Expected keyword !meta_data at line 3"
    );

    line = {};
    ASSERT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "Expected keyword !meta_data at line 3"
    );
}

/**
 * @brief a metadata line has to have exactly three elements
 *
 */
TEST_F(TestMetaDataSection, numberOfArguments)
{
    size_t lineNumber = 4;

    for (size_t count = 0; count < 8; ++count)
    {
        if (count == 3)
        {
            continue;
        }

        auto line = std::vector<std::string>(count, "version");
        if (count > 0)
        {
            line[0] = "!meta_data";
        }

        ASSERT_THROW_MSG(
            _section->process(line, *_engine, lineNumber),
            exc::RstFileException,
            count == 0 ? "Expected keyword !meta_data at line 4"
                       : "Expected 3 elements for !meta_data at line 4"
        );
    }
}

/**
 * @brief an unknown metadata key throws
 *
 */
TEST_F(TestMetaDataSection, unknownKey)
{
    auto   line       = std::vector<std::string>{"!meta_data", "colour", "red"};
    size_t lineNumber = 5;

    ASSERT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "Unrecognized metadata key 'colour' at line 5"
    );
}

/**
 * @brief the same metadata key twice throws and keeps the first value
 *
 */
TEST_F(TestMetaDataSection, duplicateKey)
{
    size_t lineNumber = 1;
    auto   line = std::vector<std::string>{"!meta_data", "version", "v1.0.0"};
    _section->process(line, *_engine, lineNumber);

    lineNumber = 2;
    line       = {"!meta_data", "version", "v2.0.0"};
    ASSERT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "Duplicate metadata key 'version' at line 2"
    );

    EXPECT_EQ(
        _settings.generalSettings.getRestartVersion(),
        SemVer::from_string("v1.0.0")
    );
}

/**
 * @brief an invalid version string is rejected and does not set the version
 *
 */
TEST_F(TestMetaDataSection, invalidVersion)
{
    for (const auto *version : {"abc", "1.2", "v1.2", "v1.2.x", "v1.2.3+"})
    {
        // a fresh section, as the section remembers parsed keys
        input::restartFile::MetaDataSection section(_settings.generalSettings);

        auto line = std::vector<std::string>{"!meta_data", "version", version};
        size_t lineNumber = 9;

        ASSERT_THROW_MSG(
            section.process(line, *_engine, lineNumber),
            exc::RstFileException,
            "Invalid version '" + std::string(version) +
                "' at line 9, expected format 'vMAJOR.MINOR.PATCH'"
        );
        EXPECT_FALSE(isValid(_settings.generalSettings.getRestartVersion()));
    }
}
