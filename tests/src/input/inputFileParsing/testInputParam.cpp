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

#include <gtest/gtest.h>   // for TestInfo (ptr only), EXPECT_EQ

#include <cstdint>
#include <string>
#include <vector>

#include "exceptions.hpp"         // for InputFileException
#include "gtest/gtest.h"          // for Message, TestPartResult
#include "inputParam.hpp"         // for InputKey, InputRegistry, KeyMetadata
#include "mstd/enum.hpp"          // for MSTD_ENUM
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG

using namespace input;

// A small MSTD_ENUM used purely to exercise the has_enum_meta<T> path of
// Converter<T> / InputKey<T> -- not a real project enum.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define TESTENUM_LIST(X) X(mm) X(qm) X(md)
MSTD_ENUM(TestJobType, std::uint8_t, TESTENUM_LIST);

/**
 * @brief tests Converter<double>::tryParse
 *
 */
TEST(TestConverter, tryParseDouble)
{
    EXPECT_EQ(Converter<double>::tryParse("1.5"), 1.5);
    EXPECT_EQ(Converter<double>::tryParse("not_a_number"), std::nullopt);
    EXPECT_EQ(Converter<double>::tryParse("1.5garbage"), std::nullopt);
}

/**
 * @brief tests Converter<bool>::tryParse
 *
 */
TEST(TestConverter, tryParseBool)
{
    EXPECT_EQ(Converter<bool>::tryParse("TRUE"), true);
    EXPECT_EQ(Converter<bool>::tryParse("ON"), true);
    EXPECT_EQ(Converter<bool>::tryParse("FALSE"), false);
    EXPECT_EQ(Converter<bool>::tryParse("OFF"), false);
    EXPECT_EQ(Converter<bool>::tryParse("maybe"), std::nullopt);
}

/**
 * @brief tests Converter<T> generic integral fallback
 *
 */
TEST(TestConverter, tryParseIntegral)
{
    EXPECT_EQ(Converter<size_t>::tryParse("42"), size_t{42});
    EXPECT_EQ(Converter<int>::tryParse("-7"), -7);
    EXPECT_EQ(Converter<size_t>::tryParse("not_a_number"), std::nullopt);
    EXPECT_EQ(Converter<size_t>::tryParse("42abc"), std::nullopt);
}

/**
 * @brief tests Converter<T> for MSTD_ENUM types via generated from_string
 *
 */
TEST(TestConverter, tryParseEnum)
{
    EXPECT_EQ(Converter<TestJobType>::tryParse("mm"), TestJobType::mm);
    EXPECT_EQ(Converter<TestJobType>::tryParse("qm"), TestJobType::qm);
    EXPECT_EQ(Converter<TestJobType>::tryParse("bogus"), std::nullopt);
}

/**
 * @brief tests that an unset key with a default resolves value() to the
 * default, and isSet() is false until explicitly parsed
 *
 */
TEST(TestInputKey, defaultValueBeforeParsing)
{
    InputKey<double> key(
        KeyMetadata{
            .name        = "timestep",
            .title       = "Timestep",
            .description = "",
            .unit        = ""
        },
        1.0
    );

    EXPECT_FALSE(key.isSet());
    EXPECT_EQ(key.value(), 1.0);
    EXPECT_FALSE(key.explicitValue().has_value());
    EXPECT_EQ(key.defaultValue(), 1.0);
}

/**
 * @brief tests that parsing sets the explicit value without touching the
 * stored default
 *
 */
TEST(TestInputKey, explicitValueOverridesDefault)
{
    InputKey<double> key(
        KeyMetadata{
            .name        = "timestep",
            .title       = "Timestep",
            .description = "",
            .unit        = ""
        },
        1.0
    );

    key.parse({"timestep", "=", "0.5"}, 1);

    EXPECT_TRUE(key.isSet());
    EXPECT_EQ(key.value(), 0.5);
    EXPECT_EQ(key.explicitValue(), 0.5);
    EXPECT_EQ(key.defaultValue(), 1.0);   // untouched
}

/**
 * @brief tests that value() throws when neither an explicit value nor a
 * default exists
 *
 */
TEST(TestInputKey, valueThrowsWithoutDefaultOrExplicit)
{
    InputKey<TestJobType> key(
        KeyMetadata{
            .name        = "jobtype",
            .title       = "Job Type",
            .description = "",
            .unit        = ""
        }
    );

    EXPECT_FALSE(key.isSet());
    EXPECT_THROW(const auto _ = key.value(), std::logic_error);
    EXPECT_FALSE(key.tryValue().has_value());
}

/**
 * @brief tests that an invalid token throws InputFileException with the
 * generated "expected one of ..." message
 *
 */
TEST(TestInputKey, invalidEnumTokenThrows)
{
    InputKey<TestJobType> key(
        KeyMetadata{
            .name        = "jobtype",
            .title       = "Job Type",
            .description = "",
            .unit        = ""
        }
    );

    EXPECT_THROW_MSG(
        key.parse({"jobtype", "=", "bogus"}, 3),
        exc::InputFileException,
        "Invalid value \"bogus\" for key \"jobtype\" at line 3 in input "
        "file. Possible options are: mm, qm, md"
    );
}

/**
 * @brief tests that a per-key errorMessage override replaces the generated
 * message
 *
 */
TEST(TestInputKey, customErrorMessageOverridesGenerated)
{
    InputKey<TestJobType> key(
        KeyMetadata{
            .name         = "jobtype",
            .title        = "Job Type",
            .description  = "",
            .unit         = "",
            .errorMessage = "Unrecognized job type"
        }
    );

    EXPECT_THROW_MSG(
        key.parse({"jobtype", "=", "bogus"}, 4),
        exc::InputFileException,
        "Unrecognized job type at line 4 in input file"
    );
}

/**
 * @brief tests the allowed-values restriction rejects an otherwise valid
 * value that is outside the given subset
 *
 */
TEST(TestInputKey, allowedSubsetRejectsOutOfRangeValue)
{
    InputKey<TestJobType> key(
        KeyMetadata{
            .name        = "jobtype",
            .title       = "Job Type",
            .description = "",
            .unit        = ""
        },
        std::nullopt,
        std::vector<TestJobType>{TestJobType::mm, TestJobType::qm}
    );

    key.parse({"jobtype", "=", "mm"}, 1);
    EXPECT_EQ(key.value(), TestJobType::mm);

    EXPECT_THROW(key.parse({"jobtype", "=", "md"}, 2), exc::InputFileException);
}

/**
 * @brief tests a customParser alias resolving a token the generated
 * from_string doesn't know about
 *
 */
TEST(TestInputKey, customParserAliasResolvesUnknownToken)
{
    InputKey<TestJobType>::CustomParser aliasParser =
        [](std::string_view raw) -> std::optional<TestJobType>
    {
        if (raw == "molecular_dynamics")
            return TestJobType::md;
        return TestJobTypeMeta::from_string(raw);
    };

    InputKey<TestJobType> key(
        KeyMetadata{
            .name        = "jobtype",
            .title       = "Job Type",
            .description = "",
            .unit        = ""
        },
        std::nullopt,
        std::nullopt,
        aliasParser
    );

    key.parse({"jobtype", "=", "molecular_dynamics"}, 1);
    EXPECT_EQ(key.value(), TestJobType::md);

    key.parse({"jobtype", "=", "qm"}, 2);
    EXPECT_EQ(key.value(), TestJobType::qm);
}

/**
 * @brief tests that onSet is invoked with the parsed value
 *
 */
TEST(TestInputKey, onSetCallbackInvoked)
{
    double captured = 0.0;

    InputKey<double> key(
        KeyMetadata{
            .name        = "timestep",
            .title       = "Timestep",
            .description = "",
            .unit        = ""
        },
        std::nullopt,
        std::nullopt,
        nullptr,
        [&captured](const double &v) { captured = v; }
    );

    key.parse({"timestep", "=", "2.5"}, 1);
    EXPECT_EQ(captured, 2.5);
}

/**
 * @brief tests describe() for an unset key with a default, an explicit
 * value, a unit, and an allowed subset
 *
 */
TEST(TestInputKey, describeReflectsState)
{
    InputKey<double> key(
        KeyMetadata{
            .name        = "timestep",
            .title       = "Timestep",
            .description = "Integration timestep for the MD run",
            .unit        = "fs"
        },
        1.0
    );

    EXPECT_EQ(
        key.describe(),
        "timestep (Timestep): Integration timestep for the MD run = 1 "
        "(default) [fs]"
    );

    key.parse({"timestep", "=", "0.5"}, 1);

    EXPECT_EQ(
        key.describe(),
        "timestep (Timestep): Integration timestep for the MD run = 0.5 "
        "[fs]"
    );
}

/**
 * @brief tests describe() includes the allowed-values list when present
 *
 */
TEST(TestInputKey, describeIncludesAllowedList)
{
    InputKey<TestJobType> key(
        KeyMetadata{
            .name        = "jobtype",
            .title       = "Job Type",
            .description = "",
            .unit        = ""
        },
        std::nullopt,
        std::vector<TestJobType>{TestJobType::mm, TestJobType::qm}
    );

    EXPECT_EQ(key.describe(), "jobtype (Job Type) = <unset> [allowed: mm, qm]");
}

/**
 * @brief tests InputRegistry: registration, dispatch by key name, unknown
 * key error, and describeAll()
 *
 */
TEST(TestInputRegistry, registerParseAndDescribeAll)
{
    InputRegistry registry;

    auto &timestepKey = registry.registerKey<double>(
        KeyMetadata{
            .name        = "timestep",
            .title       = "Timestep",
            .description = "",
            .unit        = "fs"
        },
        1.0
    );

    auto &jobTypeKey = registry.registerKey<TestJobType>(KeyMetadata{
        .name        = "jobtype",
        .title       = "Job Type",
        .description = "",
        .unit        = ""
    });

    registry.parseLine({"timestep", "=", "0.5"}, 1);
    registry.parseLine({"jobtype", "=", "qm"}, 2);

    EXPECT_EQ(timestepKey.value(), 0.5);
    EXPECT_EQ(jobTypeKey.value(), TestJobType::qm);

    EXPECT_EQ(registry.get<double>("timestep").value(), 0.5);

    const auto descriptions = registry.describeAll();
    EXPECT_EQ(descriptions.size(), size_t{2});
}

/**
 * @brief tests that an unknown key throws InputFileException
 *
 */
TEST(TestInputRegistry, unknownKeyThrows)
{
    InputRegistry registry;

    EXPECT_THROW_MSG(
        registry.parseLine({"bogus_key", "=", "1"}, 5),
        exc::InputFileException,
        "Unknown key \"bogus_key\" at line 5 in input file"
    );
}
